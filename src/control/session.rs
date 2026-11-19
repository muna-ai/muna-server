/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! Persist a fencing epoch beside the resource cache. A new server process
//! must outrank the previous one even if the node clock moves backwards.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

pub(super) fn next(cache: &Path) -> io::Result<u64> {
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    let seed = now.as_nanos().try_into()
        .map_err(|_| io::Error::other("heartbeat epoch overflow"))?;
    allocate(cache, seed)
}

fn allocate(cache: &Path, seed: u64) -> io::Result<u64> {
    fs::create_dir_all(cache)?;
    // Lock a separate inode: atomic replacement of the epoch file must not
    // release exclusion between two processes starting against the same cache.
    let lock = OpenOptions::new().create(true).truncate(false).write(true)
        .open(cache.join(".heartbeat-session.lock"))?;
    lock.lock()?;
    let path = cache.join(".heartbeat-session");
    let previous = match fs::read_to_string(&path) {
        Ok(text) => text.trim().parse::<u64>()
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid heartbeat session epoch"))?,
        Err(error) if error.kind() == io::ErrorKind::NotFound => 0,
        Err(error) => return Err(error),
    };
    let session = seed.max(previous.checked_add(1)
        .ok_or_else(|| io::Error::other("heartbeat epoch exhausted"))?);
    let temporary = cache.join(".heartbeat-session.tmp");
    let mut file = File::create(&temporary)?;
    writeln!(file, "{session}")?;
    file.sync_all()?;
    fs::rename(temporary, path)?;
    #[cfg(unix)]
    File::open(cache)?.sync_all()?;
    // Dropping the lock file releases the process-shared lock.
    Ok(session)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    fn cache() -> std::path::PathBuf {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        std::env::temp_dir().join(format!("muna-heartbeat-session-{}-{}",
            std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)))
    }

    #[test]
    fn restart_epoch_increases_even_when_the_clock_moves_backwards() {
        let cache = cache();
        assert_eq!(allocate(&cache, 100).unwrap(), 100);
        assert_eq!(allocate(&cache, 99).unwrap(), 101);
        assert_eq!(allocate(&cache, 0).unwrap(), 102);
        assert_eq!(allocate(&cache, 200).unwrap(), 200);
        fs::remove_dir_all(cache).unwrap();
    }

    #[test]
    fn concurrent_starts_allocate_distinct_epochs() {
        let cache = cache();
        let workers: Vec<_> = (0..8).map(|_| {
            let cache = cache.clone();
            std::thread::spawn(move || allocate(&cache, 100).unwrap())
        }).collect();
        let mut sessions: Vec<_> = workers.into_iter().map(|worker| worker.join().unwrap()).collect();
        sessions.sort_unstable();
        assert_eq!(sessions, (100..108).collect::<Vec<_>>());
        fs::remove_dir_all(cache).unwrap();
    }
}
