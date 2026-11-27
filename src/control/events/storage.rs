/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! One atomically committed file per event. A crash during acknowledgement
//! cleanup can replay a committed event; control-plane deduplication makes
//! that safe. Never discard unreadable committed files silently.

use std::fs::{self, File};
use std::io::{self, Write};
use std::path::Path;

use super::ModelLoadEvent;

pub(super) fn read(path: &Path) -> io::Result<Vec<ModelLoadEvent>> {
    fs::create_dir_all(path)?;
    let mut events = Vec::new();
    for entry in fs::read_dir(path)? {
        let path = entry?.path();
        if path.extension().is_none_or(|extension| extension != "json") { continue; }
        let event: ModelLoadEvent = serde_json::from_slice(&fs::read(&path)?)?;
        // Only UUID filenames can ever be used for writes or removal.
        let id = uuid::Uuid::parse_str(&event.id).map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        if id.to_string() != event.id || path.file_stem().and_then(|stem| stem.to_str()) != Some(&event.id) {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "load event filename does not match its ID"));
        }
        events.push(event);
    }
    Ok(events)
}

pub(super) fn write(path: &Path, event: &ModelLoadEvent) -> io::Result<()> {
    let temporary = path.join(format!("{}.tmp", event.id));
    let mut file = File::create(&temporary)?;
    file.write_all(&serde_json::to_vec(event)?)?;
    file.sync_all()?;
    fs::rename(temporary, path.join(format!("{}.json", event.id)))?;
    sync_directory(path)
}

pub(super) fn remove(path: &Path, id: &str) -> io::Result<()> {
    match fs::remove_file(path.join(format!("{id}.json"))) {
        Ok(()) => {},
        Err(error) if error.kind() == io::ErrorKind::NotFound => {},
        Err(error) => return Err(error),
    }
    sync_directory(path)
}

fn sync_directory(path: &Path) -> io::Result<()> {
    #[cfg(unix)]
    File::open(path)?.sync_all()?;
    Ok(())
}
