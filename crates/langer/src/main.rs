//! Entry point: configuration, logging, model loading and graceful shutdown.

use std::io::IsTerminal;
use std::net::{Ipv4Addr, SocketAddr};
use std::thread;
use std::time::Instant;

use anyhow::Context;
use axum::serve::ListenerExt;
use clap::Parser;
use langer::{Detectors, Limits};
use langer_mediapipe::Detector;
use lingua::LanguageDetectorBuilder;
use tracing::Level;
use tracing_subscriber::EnvFilter;

/// Language detection HTTP service.
#[derive(Debug, Parser)]
#[command(version, about)]
struct Config {
    /// Port to listen on.
    #[arg(long, env = "PORT", default_value_t = 3000)]
    port: u16,
    /// Largest accepted request body, in bytes.
    #[arg(long, env = "MAX_BODY_BYTES", default_value_t = Limits::default().body_bytes)]
    max_body_bytes: usize,
}

fn main() -> anyhow::Result<()> {
    let config = Config::parse();
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_env("LOG_LEVEL").unwrap_or_else(|_| EnvFilter::new("info")))
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

async fn serve(port: u16, detectors: Detectors, limits: Limits) -> anyhow::Result<()> {
    let addr = SocketAddr::from((Ipv4Addr::UNSPECIFIED, port));
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("cannot listen on {addr}"))?
        .tap_io(|tcp| {
            // Responses are small and written at once; don't let Nagle delay them.
            if let Err(e) = tcp.set_nodelay(true) {
                tracing::event!(name: "server.nodelay.failure", Level::WARN, error.message = %e, "cannot set TCP_NODELAY");
            }
        });
    tracing::event!(name: "server.listen.success", Level::INFO, server.address = %addr, "listening");
    axum::serve(listener, langer::router(detectors, limits))
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    tracing::event!(name: "server.shutdown.success", Level::INFO, "stopped");
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
