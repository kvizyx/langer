//! Entry point: configuration, logging, model loading and graceful shutdown.

use std::io::IsTerminal;
use std::net::{Ipv4Addr, Ipv6Addr, SocketAddr};
use std::num::NonZeroUsize;
use std::thread;
use std::time::Instant;

use anyhow::Context;
use axum::serve::ListenerExt;
use clap::Parser;
use langer::{Detectors, Limits};
use langer_mediapipe::Detector;
use lingua::LanguageDetectorBuilder;
use tokio::net::TcpListener;
use tracing::Level;
use tracing_subscriber::EnvFilter;

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
    #[arg(long, env = "MAX_BODY_BYTES", default_value_t = Limits::default().body_bytes)]
    max_body_bytes: NonZeroUsize,
    /// Most `lingua` detections running or queued at once; more get 503. Defaults to four per core.
    #[arg(long, env = "LINGUA_MAX_IN_FLIGHT", default_value_t = Limits::default().lingua_in_flight)]
    lingua_max_in_flight: NonZeroUsize,
}

fn main() -> anyhow::Result<()> {
    let config = Config::parse();
    let filter =
        EnvFilter::try_new(&config.log_level).with_context(|| format!("invalid LOG_LEVEL {:?}", config.log_level))?;
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_ansi(std::io::stdout().is_terminal())
        .init();

    let detectors = load_detectors()?;

    // Lingua runs on blocking threads; more of them than cores only adds memory and
    // context switches for this CPU-bound work. Respects cgroup CPU limits.
    let cores = thread::available_parallelism().map_or(1, usize::from);
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .max_blocking_threads(cores)
        .build()?
        .block_on(serve(
            config.port,
            detectors,
            Limits {
                body_bytes: config.max_body_bytes,
                lingua_in_flight: config.lingua_max_in_flight,
            },
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

async fn serve(port: u16, detectors: Detectors, limits: Limits) -> anyhow::Result<()> {
    let (listener, addr) = bind(port).await?;

    let listener = listener.tap_io(|tcp| {
        // Responses are small and written at once; don't let Nagle delay them.
        if let Err(e) = tcp.set_nodelay(true) {
            tracing::event!(name: "server.nodelay.failure", Level::WARN, error.message = %e, "cannot set TCP_NODELAY");
        }
    });

    tracing::event!(name: "server.listen.success", Level::INFO, server.address = %addr, "server is listening");
    axum::serve(listener, langer::router(detectors, limits))
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    tracing::event!(name: "server.shutdown.success", Level::INFO, "server is stopped");

    Ok(())
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
