//! Entry point: configuration, logging, model loading and graceful shutdown.

use std::io::IsTerminal;
use std::net::{Ipv4Addr, Ipv6Addr, SocketAddr};
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Context;
use axum::serve::ListenerExt;
use clap::Parser;
use langer::{Detectors, Limits, LinguaLimits, Service};
use langer_mediapipe::Detector;
use lingua::LanguageDetectorBuilder;
use metrics_exporter_prometheus::PrometheusHandle;
use tokio::net::TcpListener;
use tokio::sync::Notify;
use tracing::Level;
use tracing_subscriber::EnvFilter;

/// How often the metrics recorder drains histogram data.
const METRICS_UPKEEP: Duration = Duration::from_secs(5);

/// Language detection HTTP service.
#[derive(Debug, Parser)]
#[command(version, about)]
struct Config {
    /// Port to listen on.
    #[arg(long, env = "PORT", default_value_t = 3000)]
    port: u16,
    /// Log filter: a level (`warn`) or `tracing` directives (`info,langer=debug`).
    #[arg(long, env = "LOG_LEVEL", default_value = "info")]
    log_level: String,
    /// Largest accepted request body, in bytes.
    #[arg(long, env = "MAX_BODY_BYTES", default_value_t = langer::DEFAULT_BODY_LIMIT)]
    max_body_bytes: NonZeroUsize,
    /// Seconds a request may take, from its headers to the response; slower ones get 408.
    #[arg(long, env = "REQUEST_TIMEOUT_SECONDS", default_value_t = langer::DEFAULT_REQUEST_TIMEOUT.as_secs())]
    request_timeout_seconds: u64,
    /// Most `lingua` detections running or waiting for a thread at once; more wait for a free slot.
    /// Defaults to four per core.
    #[arg(long, env = "LINGUA_MAX_IN_FLIGHT", default_value_t = langer::default_lingua_in_flight())]
    lingua_max_in_flight: NonZeroUsize,
    /// Milliseconds a `lingua` detection may wait for a free slot before it gets 503; 0 refuses
    /// at once.
    #[arg(
        long,
        env = "LINGUA_QUEUE_TIMEOUT_MILLISECONDS",
        default_value_t = langer::DEFAULT_LINGUA_QUEUE_TIMEOUT.as_millis().try_into().unwrap_or(u64::MAX)
    )]
    lingua_queue_timeout_milliseconds: u64,
    /// Seconds between a stop signal and closing the listener, while `GET /health/ready` answers
    /// 503 and requests are still served.
    #[arg(long, env = "SHUTDOWN_DELAY_SECONDS", default_value_t = 5)]
    shutdown_delay_seconds: u64,
}

fn main() -> anyhow::Result<()> {
    let config = Config::parse();
    let filter =
        EnvFilter::try_new(&config.log_level).with_context(|| format!("invalid LOG_LEVEL {:?}", config.log_level))?;
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_ansi(std::io::stdout().is_terminal())
        .init();

    // Before the service, which records its first metrics when it is created.
    let metrics = langer::install_metrics().context("cannot install the metrics recorder")?;
    let lingua = LinguaLimits {
        in_flight: config.lingua_max_in_flight,
        queue_timeout: Duration::from_millis(config.lingua_queue_timeout_milliseconds),
    };
    let service = Service::new(load_detectors()?, lingua);
    let limits = Limits {
        body: config.max_body_bytes,
        request_timeout: Duration::from_secs(config.request_timeout_seconds),
    };

    // Lingua runs on blocking threads; more of them than cores only adds memory and
    // context switches for this CPU-bound work. Respects cgroup CPU limits.
    let cores = thread::available_parallelism().map_or(1, usize::from);
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .max_blocking_threads(cores)
        .build()?
        .block_on(serve(
            config.port,
            service,
            limits,
            metrics,
            Duration::from_secs(config.shutdown_delay_seconds),
        ))
}

fn load_detectors() -> anyhow::Result<Detectors> {
    let started = Instant::now();
    let mediapipe = Detector::new().context("cannot load the MediaPipe language detector")?;
    tracing::event!(
        name: "model.mediapipe.loaded",
        Level::INFO,
        model.size_kib = mediapipe.heap_size() >> 10,
        duration_us = u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX),
        "loaded MediaPipe language detector",
    );

    // Lingua's models are memory-mapped from the executable on first use; nothing to preload.
    let lingua = LanguageDetectorBuilder::from_all_languages().build();

    Ok(Detectors::new(mediapipe, lingua))
}

/// Listens on every IPv6 and IPv4 address: an IPv6 socket accepts IPv4 too on Linux by default
/// (`net.ipv6.bindv6only = 0`). Falls back to IPv4 only where the kernel has no IPv6.
async fn bind(port: u16) -> anyhow::Result<(TcpListener, SocketAddr)> {
    let dual_stack = SocketAddr::from((Ipv6Addr::UNSPECIFIED, port));

    match TcpListener::bind(dual_stack).await {
        Ok(listener) => Ok((listener, dual_stack)),
        Err(e) => {
            tracing::event!(
                name: "server.ipv6.failure",
                Level::WARN,
                error.message = %e,
                "cannot listen on IPv6, listening on IPv4 only",
            );

            let ipv4 = SocketAddr::from((Ipv4Addr::UNSPECIFIED, port));
            let listener = TcpListener::bind(ipv4)
                .await
                .with_context(|| format!("cannot listen on {ipv4}"))?;

            Ok((listener, ipv4))
        }
    }
}

async fn serve(
    port: u16,
    service: Service,
    limits: Limits,
    metrics: PrometheusHandle,
    shutdown_delay: Duration,
) -> anyhow::Result<()> {
    let (listener, addr) = bind(port).await?;

    let listener = listener.tap_io(|tcp| {
        // Responses are small and written at once; don't let Nagle delay them.
        if let Err(e) = tcp.set_nodelay(true) {
            tracing::event!(name: "server.nodelay.failure", Level::WARN, error.message = %e, "cannot set TCP_NODELAY");
        }
    });

    tracing::event!(name: "server.listen.success", Level::INFO, server.address = %addr, "server is listening");

    // Without the exporter's own HTTP listener, nothing else drains histogram data between scrapes.
    let upkeep = metrics.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(METRICS_UPKEEP);
        loop {
            interval.tick().await;
            upkeep.run_upkeep();
        }
    });

    let closed = Arc::new(Notify::new());
    let server = axum::serve(listener, langer::router(service.clone(), limits, metrics))
        .with_graceful_shutdown(shut_down(service, shutdown_delay, Arc::clone(&closed)))
        .into_future();

    // Requests in flight end within the request timeout, but a connection that never sends a
    // full request has no timeout and would keep the server from stopping.
    let drained = async {
        closed.notified().await;
        tokio::time::sleep(limits.request_timeout).await;
    };

    tokio::select! {
        result = server => result?,
        () = drained => {
            tracing::event!(
                name: "server.shutdown.timeout",
                Level::WARN,
                "connections still open after the request timeout, closing them",
            );
        }
    }

    tracing::event!(name: "server.shutdown.success", Level::INFO, "server is stopped");

    Ok(())
}

/// Resolves when the server should stop accepting connections: a stop signal, then
/// `delay` of reporting not ready while still serving, so that clients move to other instances.
/// Notifies `closed` when it resolves.
async fn shut_down(service: Service, delay: Duration, closed: Arc<Notify>) {
    shutdown_signal().await;

    service.shut_down();
    tracing::event!(
        name: "server.shutdown.start",
        Level::INFO,
        shutdown.delay_seconds = delay.as_secs(),
        "server is shutting down",
    );
    tokio::time::sleep(delay).await;

    closed.notify_one();
}

/// Resolves on Ctrl+C or SIGTERM (sent by Kubernetes before killing the pod).
async fn shutdown_signal() {
    let ctrl_c = async {
        if let Err(e) = tokio::signal::ctrl_c().await {
            tracing::event!(name: "server.signal.failure", Level::WARN, error.message = %e, "cannot listen for Ctrl+C");
            std::future::pending::<()>().await;
        }
    };

    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut signal) => {
                signal.recv().await;
            }
            Err(e) => {
                tracing::event!(name: "server.signal.failure", Level::WARN, error.message = %e, "cannot listen for SIGTERM");
                std::future::pending::<()>().await;
            }
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => {},
        () = terminate => {},
    }
}
