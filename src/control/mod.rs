/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! Node-side control-plane integration: heartbeat, KV event relay, and the
//! node-control wire protocol. Everything here runs only when the server is
//! started with `--control-plane-url`.

pub(crate) mod heartbeat;
pub(crate) mod kv_relay;
pub(crate) mod protocol;
