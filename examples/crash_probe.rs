/*
*   Muna
*   Copyright © 2026 NatML Inc. All rights reserved.
*/

// Deliberate failures live in an opt-in example, never a server CLI endpoint.
#[path = "../src/crash_reporting.rs"]
mod crash_reporting;

#[inline(never)]
fn rust_panic_probe() {
    panic!("deliberate panic for crash-reporting verification");
}

#[cfg(target_os = "linux")]
#[inline(never)]
fn fatal_probe(signal: i32) {
    unsafe { libc::raise(signal) };
    std::hint::black_box(signal);
}

#[cfg(target_os = "linux")]
#[inline(never)]
fn native_library_probe() {
    let path = std::env::args()
        .nth(2)
        .expect("expected native probe library path");
    unsafe {
        let library = libloading::Library::new(path).unwrap();
        let crash: libloading::Symbol<unsafe extern "C" fn()> =
            library.get(b"native_crash_probe").unwrap();
        crash();
        std::hint::black_box(&library);
    }
}

fn main() {
    crash_reporting::run_reporter_if_requested();
    crash_reporting::install();
    crash_reporting::register_thread();
    match std::env::args().nth(1).as_deref() {
        Some("panic") => rust_panic_probe(),
        Some("caught-panic") => {
            let _ = std::panic::catch_unwind(rust_panic_probe);
        }
        #[cfg(target_os = "linux")]
        Some("segv") => fatal_probe(libc::SIGSEGV),
        #[cfg(target_os = "linux")]
        Some("abort") => fatal_probe(libc::SIGABRT),
        #[cfg(target_os = "linux")]
        Some("worker-segv") => {
            std::thread::Builder::new()
                .name("native-crash-probe".into())
                .spawn(|| {
                    crash_reporting::register_thread();
                    fatal_probe(libc::SIGSEGV);
                })
                .unwrap()
                .join()
                .unwrap();
        }
        #[cfg(target_os = "linux")]
        Some("helper-dead") => {
            let pid = std::process::id();
            let children =
                std::fs::read_to_string(format!("/proc/self/task/{pid}/children")).unwrap();
            let helper: i32 = children.split_whitespace().next().unwrap().parse().unwrap();
            unsafe {
                libc::kill(helper, libc::SIGKILL);
                libc::waitpid(helper, std::ptr::null_mut(), 0);
            }
            fatal_probe(libc::SIGSEGV);
        }
        #[cfg(target_os = "linux")]
        Some("native-segv") => native_library_probe(),
        Some("normal") => {}
        _ => panic!("expected panic, caught-panic, segv, abort, worker-segv, or normal"),
    }
}
