/*
*   Muna
*   Copyright © 2026 NatML Inc. All rights reserved.
*/

use std::backtrace::Backtrace;
use std::io::Write;

#[cfg(target_os = "linux")]
extern "C" {
    fn muna_crash_install() -> i32;
    fn muna_crash_register_thread() -> i32;
    fn muna_crash_reporter(fd: i32) -> i32;
}

/// Dispatch before creating the async runtime or initializing the normal server.
/// The helper is a clean exec of this binary with one inherited Unix socket.
pub fn run_reporter_if_requested() {
    #[cfg(target_os = "linux")]
    {
        let args: Vec<_> = std::env::args_os().collect();
        if args.len() == 3 && args[1] == "--muna-crash-reporter" {
            let fd = args[2].to_str().and_then(|s| s.parse::<i32>().ok());
            let code = match fd.filter(|fd| *fd >= 3) {
                Some(fd) => unsafe { muna_crash_reporter(fd) },
                None => 2,
            };
            std::process::exit(code);
        }
    }
}

/// Install exactly once, before Tokio or any model worker threads start.
pub fn install() {
    std::panic::set_hook(Box::new(|info| {
        let thread = std::thread::current();
        let backtrace = Backtrace::force_capture();
        let unix_time_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|time| time.as_millis())
            .unwrap_or_default();
        // Bypass tracing filters and ignore I/O errors: panicking in this hook
        // would abort before the original panic's report completes.
        let _ = writeln!(
            std::io::stderr().lock(),
            "\n=== MUNA RUST PANIC ===\nversion={} unix_time_ms={} pid={} thread={:?} name={:?}\n{info}\n{backtrace:#}\n=== END MUNA RUST PANIC ===",
            env!("CARGO_PKG_VERSION"),
            unix_time_ms,
            std::process::id(),
            thread.id(),
            thread.name()
        );
    }));
    #[cfg(target_os = "linux")]
    {
        let status = unsafe { muna_crash_install() };
        if status != 0 {
            eprintln!("MUNA: fatal-signal crash reporter unavailable (errno={status}); Rust panic reporting is enabled");
        }
    }
}

/// Alternate signal stacks are thread-local; call on Tokio worker startup too.
pub fn register_thread() {
    #[cfg(target_os = "linux")]
    {
        let status = unsafe { muna_crash_register_thread() };
        if status != 0 {
            let _ = writeln!(
                std::io::stderr(),
                "MUNA: could not install alternate signal stack (errno={status})"
            );
        }
    }
}
