/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! Positional writes keep every range in one file, without a shared seek cursor.

use std::fs::File;
use std::sync::Arc;

pub(super) async fn write(file: Arc<File>, bytes: Vec<u8>, offset: u64) -> Result<u64, String> {
    let count = bytes.len() as u64;
    tokio::task::spawn_blocking(move || write_all_at(&file, &bytes, offset))
        .await
        .map_err(|error| format!("Resource write task failed: {error}"))?
        .map_err(|error| format!("Failed to write resource: {error}"))?;
    Ok(count)
}

#[cfg(unix)]
fn write_all_at(file: &File, bytes: &[u8], offset: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.write_all_at(bytes, offset)
}

#[cfg(windows)]
fn write_all_at(file: &File, mut bytes: &[u8], mut offset: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !bytes.is_empty() {
        match file.seek_write(bytes, offset) {
            Ok(0) => return Err(std::io::ErrorKind::WriteZero.into()),
            Ok(count) => {
                bytes = &bytes[count..];
                offset += count as u64;
            }
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    Ok(())
}
