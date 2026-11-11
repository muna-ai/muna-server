/*
*   Muna
*   Copyright © 2026 NatML Inc. All rights reserved.
*/

use std::{env, fs, path::PathBuf, process::Command};

fn run(command: &mut Command) {
    let status = command
        .status()
        .expect("could not execute native build tool");
    assert!(
        status.success(),
        "native crash reporter build failed: {command:?}"
    );
}

fn main() {
    println!("cargo:rerun-if-changed=native");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux") {
        return;
    }
    let build = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("crash-reporter");
    let source = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("native");
    run(Command::new("cmake")
        .arg("-S")
        .arg(source)
        .arg("-B")
        .arg(&build)
        .arg("-DCMAKE_BUILD_TYPE=RelWithDebInfo"));
    run(Command::new("cmake")
        .arg("--build")
        .arg(&build)
        .arg("--target")
        .arg("muna_crash")
        .arg("--parallel")
        .arg(env::var("NUM_JOBS").unwrap_or_else(|_| "2".into())));
    let libraries = fs::read_to_string(build.join("native-libraries.txt")).unwrap();
    for library in libraries.lines().filter(|line| !line.is_empty()) {
        let path = PathBuf::from(library);
        let name = path.file_name().unwrap().to_str().unwrap();
        assert!(
            name.starts_with("lib") && name.ends_with(".a"),
            "expected static archive: {library}"
        );
        println!(
            "cargo:rustc-link-search=native={}",
            path.parent().unwrap().display()
        );
        println!("cargo:rustc-link-lib=static={}", &name[3..name.len() - 2]);
    }
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=dl");
    println!("cargo:rustc-link-arg=-Wl,--build-id=sha1");
    // Resolve PLT entries before a signal can interrupt the dynamic loader.
    println!("cargo:rustc-link-arg=-Wl,-z,now");
}
