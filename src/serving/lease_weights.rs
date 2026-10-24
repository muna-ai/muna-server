/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! Local device-lease weight solver.
//!
//! Policy and weights are split. The control plane sends a slow-moving
//! [`LeasePolicy`] per model (value per lease-held GPU-second, and a floor
//! on its share of a contended device); this loop turns it into lease
//! weights every second from live lease counters, which the plane never
//! sees at that cadence.
//!
//! Each tick:
//! 1. Demand per (device, model) is the fraction of wall time the engine
//!    spent holding or waiting for the lease, smoothed with an EWMA.
//! 2. Each device's time is split by a fractional knapsack: floors first,
//!    then remaining time to the highest-value demand, then leftovers by
//!    value.
//! 3. Shares become stride weights on that device.
//!
//! Spawned unconditionally in `main`: co-resident engines arbitrate in
//! standalone mode too, where every model runs on the defaults below.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::control::protocol::LeasePolicy;
use crate::serving::lease::RegistrantStats;
use crate::state::AppState;

/// Cadence of the solver loop.
const WEIGHT_INTERVAL: Duration = Duration::from_secs(1);
/// Weight per unit of share. Shares on a device sum to about 1, so weights
/// on a device sum to about this.
const WEIGHT_SCALE: f64 = 1000.0;
/// EWMA time constant for demand, in seconds.
const DEMAND_TAU_S: f64 = 5.0;
/// Demand assumed for a newly registered engine until its counters have
/// a delta: presumed busy, so a fresh load is not starved while warming.
const INITIAL_DEMAND: f64 = 1.0;
/// Value for a model without a policy. Equal for every model: the policy
/// is the only source of relative value, so two saturated co-tenants
/// without one split evenly.
const DEFAULT_VALUE: f64 = 1.0;
/// Share floor for a model without a policy.
const DEFAULT_MIN_SHARE: f64 = 0.25;

/// One registrant's input to the per-device share split.
#[derive(Clone, Copy, Debug, PartialEq)]
struct ShareInput {
    /// Smoothed fraction of device time the model wants, in [0, 1].
    demand: f64,
    /// Value per unit of device time; higher wins remaining time.
    value: f64,
    /// Floor on the share granted (capped at demand), in [0, 1).
    min_share: f64,
}

/// Fractional-knapsack share split for one device: floors first, then
/// remaining time to the highest-value demand, then leftovers by value.
/// Models of equal value split the time they compete for evenly.
fn solve_shares(entries: &[ShareInput]) -> Vec<f64> {
    const EPSILON: f64 = 1e-12;
    let mut share: Vec<f64> = entries.iter().map(|e| e.min_share.min(e.demand)).collect();
    let mut free = (1.0 - share.iter().sum::<f64>()).max(0.0);
    let mut order: Vec<usize> = (0..entries.len()).collect();
    order.sort_by(|&a, &b| entries[b].value.total_cmp(&entries[a].value));
    for tier in order.chunk_by(|&a, &b| entries[a].value == entries[b].value) {
        // Water-fill: split what's free evenly across the tier, repeating
        // while some members are capped by their demand and others want more.
        let mut wanting = tier.to_vec();
        while free > EPSILON && !wanting.is_empty() {
            let each = free / wanting.len() as f64;
            wanting.retain(|&i| {
                let grant = (entries[i].demand - share[i]).clamp(0.0, each);
                share[i] += grant;
                free -= grant;
                entries[i].demand - share[i] > EPSILON
            });
        }
    }
    let total_value: f64 = entries.iter().map(|e| e.value).sum();
    for (s, e) in share.iter_mut().zip(entries) {
        *s += free * e.value / total_value.max(f64::EPSILON);
    }
    share
}

/// Shares for every (device, label) input, solved per device.
///
/// A tensor-parallel engine holds one registrant per rank device and steps
/// in lockstep, so it can use no more than its smallest share on any of
/// them. A second pass caps each label's demand at that minimum and
/// re-solves, so the excess goes to co-tenants instead of idling.
fn plan_shares(inputs: &[(i32, String, ShareInput)]) -> Vec<f64> {
    let mut by_device: BTreeMap<i32, Vec<usize>> = BTreeMap::new();
    for (index, (device, _, _)) in inputs.iter().enumerate() {
        by_device.entry(*device).or_default().push(index);
    }
    let solve = |demand_cap: &dyn Fn(usize) -> f64| {
        let mut shares = vec![0.0; inputs.len()];
        for indices in by_device.values() {
            let entries: Vec<ShareInput> = indices
                .iter()
                .map(|&i| ShareInput {
                    demand: inputs[i].2.demand.min(demand_cap(i)),
                    ..inputs[i].2
                })
                .collect();
            for (&i, share) in indices.iter().zip(solve_shares(&entries)) {
                shares[i] = share;
            }
        }
        shares
    };
    let first = solve(&|_| f64::INFINITY);
    let mut cap: HashMap<&str, (usize, f64)> = HashMap::new();
    for (i, (_, label, _)) in inputs.iter().enumerate() {
        let (devices, min_share) = cap.entry(label.as_str()).or_insert((0, f64::INFINITY));
        *devices += 1;
        *min_share = min_share.min(first[i]);
    }
    solve(&|i| match cap[inputs[i].1.as_str()] {
        (1, _) => f64::INFINITY,
        (_, min_share) => min_share,
    })
}

/// Stride weight for a share; never 0, which would mean free-run.
fn weight(share: f64) -> u32 {
    (share * WEIGHT_SCALE).round().max(1.0) as u32
}

/// Counter snapshot and smoothed demand for one (device, label).
struct DemandSample {
    ns_held: u64,
    ns_waited: u64,
    at: Instant,
    demand: f64,
}

/// Smoothed per-(device, label) demand derived from lease counters.
#[derive(Default)]
struct DemandTracker {
    samples: HashMap<(i32, String), DemandSample>,
}

impl DemandTracker {

    /// Fold one counter snapshot in and return each registrant's smoothed
    /// demand, in `stats` order. Registrants that vanished are dropped; a
    /// counter that went backwards (engine reloaded) restarts the sample.
    fn observe(
        &mut self,
        stats: &[RegistrantStats],
        now: Instant
    ) -> Vec<f64> {
        let mut seen = HashSet::with_capacity(stats.len());
        let demands = stats
            .iter()
            .map(|stat| {
                let key = (stat.device, stat.label.clone());
                let demand = match self.samples.get_mut(&key) {
                    Some(sample)
                        if stat.ns_held >= sample.ns_held
                            && stat.ns_waited >= sample.ns_waited =>
                    {
                        let dt = now.saturating_duration_since(sample.at).as_secs_f64();
                        if dt > 0.0 {
                            let held_ns = stat.ns_held - sample.ns_held;
                            let busy_ns = held_ns + (stat.ns_waited - sample.ns_waited);
                            let instant = (busy_ns as f64 / (dt * 1e9)).clamp(0.0, 1.0);
                            let alpha = 1.0 - (-dt / DEMAND_TAU_S).exp();
                            sample.demand += alpha * (instant - sample.demand);
                            sample.ns_held = stat.ns_held;
                            sample.ns_waited = stat.ns_waited;
                            sample.at = now;
                        }
                        sample.demand
                    },
                    _ => {
                        self.samples.insert(key.clone(), DemandSample {
                            ns_held: stat.ns_held,
                            ns_waited: stat.ns_waited,
                            at: now,
                            demand: INITIAL_DEMAND
                        });
                        INITIAL_DEMAND
                    }
                };
                seen.insert(key);
                demand
            })
            .collect();
        self.samples.retain(|key, _| seen.contains(key));
        demands
    }
}

/// The policy for a label: the control plane's when it sent one, otherwise
/// the standalone defaults. Sanitized so a malformed policy cannot produce
/// a NaN weight.
fn policy_for(
    state: &AppState,
    label: &str
) -> LeasePolicy {
    let default = LeasePolicy {
        value_per_gpu_s: DEFAULT_VALUE,
        min_share: DEFAULT_MIN_SHARE,
    };
    let Some(policy) = state.policies.get(label).map(|entry| *entry) else {
        return default;
    };
    LeasePolicy {
        value_per_gpu_s: if policy.value_per_gpu_s.is_finite() {
            policy.value_per_gpu_s.max(0.0)
        } else {
            default.value_per_gpu_s
        },
        min_share: if policy.min_share.is_finite() {
            policy.min_share.clamp(0.0, 1.0)
        } else {
            default.min_share
        },
    }
}

/// Solver loop. Weights only bind under contention (the lease is
/// work-conserving), so a lone model is unaffected; idempotent atomic
/// stores make the per-second re-assert free.
pub(crate) async fn run(state: Arc<AppState>) {
    let mut interval = tokio::time::interval(WEIGHT_INTERVAL);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut tracker = DemandTracker::default();
    loop {
        tokio::select! {
            _ = state.shutdown.cancelled() => return,
            _ = interval.tick() => {}
        }
        let stats = state.lease.stats();
        if stats.is_empty() {
            continue;
        }
        let demands = tracker.observe(&stats, Instant::now());
        let inputs: Vec<(i32, String, ShareInput)> = stats
            .iter()
            .zip(demands)
            .map(|(stat, demand)| {
                let policy = policy_for(&state, &stat.label);
                let input = ShareInput {
                    demand,
                    value: policy.value_per_gpu_s,
                    min_share: policy.min_share
                };
                (stat.device, stat.label.clone(), input)
            })
            .collect();
        for ((device, label, _), share) in inputs.iter().zip(plan_shares(&inputs)) {
            state.lease.set_weight_on(*device, label, weight(share));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(
        demand: f64,
        value: f64,
        min_share: f64
    ) -> ShareInput {
        ShareInput { demand, value, min_share }
    }

    fn assert_close(
        actual: &[f64],
        expected: &[f64]
    ) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() < 1e-9, "{actual:?} != {expected:?}");
        }
    }

    fn stats(
        device: i32,
        label: &str,
        ns_held: u64,
        ns_waited: u64
    ) -> RegistrantStats {
        RegistrantStats {
            device,
            label: label.to_string(),
            weight: 1,
            slots_held: 0,
            ns_held,
            ns_waited,
            yields: 0
        }
    }

    #[test]
    fn defaults_split_evenly() {
        let shares = solve_shares(&[
            input(1.0, DEFAULT_VALUE, DEFAULT_MIN_SHARE),
            input(1.0, DEFAULT_VALUE, DEFAULT_MIN_SHARE)
        ]);
        assert_close(&shares, &[0.5, 0.5]);
        assert_eq!(weight(shares[0]), weight(shares[1]));
    }

    #[test]
    fn equal_values_water_fill() {
        // The light model's demand is met; the two saturated ones split
        // what remains evenly.
        let shares = solve_shares(&[
            input(1.0, 1.0, 0.0),
            input(0.1, 1.0, 0.0),
            input(1.0, 1.0, 0.0)
        ]);
        assert_close(&shares, &[0.45, 0.1, 0.45]);
    }

    #[test]
    fn floors_hold_against_higher_value() {
        let shares = solve_shares(&[input(1.0, 100.0, 0.05), input(1.0, 1.0, 0.05)]);
        assert_close(&shares, &[0.95, 0.05]);
    }

    #[test]
    fn floor_is_capped_at_demand() {
        // An idle model's floor is not reserved; its unused time goes to
        // the busy model, and the leftover is split by value.
        let shares = solve_shares(&[input(1.0, 1.0, 0.25), input(0.0, 1.0, 0.25)]);
        assert_close(&shares, &[1.0, 0.0]);
    }

    #[test]
    fn leftovers_split_by_value() {
        let shares = solve_shares(&[input(0.2, 3.0, 0.0), input(0.2, 1.0, 0.0)]);
        assert_close(&shares, &[0.2 + 0.6 * 0.75, 0.2 + 0.6 * 0.25]);
    }

    #[test]
    fn lone_model_gets_the_device() {
        assert_close(&solve_shares(&[input(0.1, 1.0, 0.05)]), &[1.0]);
        assert!(solve_shares(&[]).is_empty());
    }

    #[test]
    fn zero_values_do_not_produce_nan() {
        let shares = solve_shares(&[input(0.1, 0.0, 0.0), input(0.1, 0.0, 0.0)]);
        assert!(shares.iter().all(|s| s.is_finite()));
        assert!(shares.iter().all(|&s| weight(s) >= 1));
    }

    #[test]
    fn tensor_parallel_share_is_capped_at_its_bottleneck() {
        // TP2 model A spans devices 0 and 1. On device 1 it loses to the
        // high-value C, so it can only use 5% of device 0 too; B gets the
        // rest of device 0 instead of A idling on it.
        let inputs = vec![
            (0, "A".to_string(), input(1.0, 3.0, 0.05)),
            (0, "B".to_string(), input(1.0, 1.0, 0.05)),
            (1, "A".to_string(), input(1.0, 3.0, 0.05)),
            (1, "C".to_string(), input(1.0, 10.0, 0.05))
        ];
        assert_close(&plan_shares(&inputs), &[0.05, 0.95, 0.05, 0.95]);
    }

    #[test]
    fn demand_converges_toward_busy_fraction() {
        let mut tracker = DemandTracker::default();
        let start = Instant::now();
        assert_eq!(tracker.observe(&[stats(0, "m", 0, 0)], start), vec![INITIAL_DEMAND]);
        // Busy 20% of each second (held 0.1 s + waited 0.1 s).
        let mut demand = INITIAL_DEMAND;
        for second in 1..=60u64 {
            let ns = second * 100_000_000;
            demand = tracker.observe(
                &[stats(0, "m", ns, ns)],
                start + Duration::from_secs(second)
            )[0];
        }
        assert!((demand - 0.2).abs() < 1e-3, "demand {demand}");
    }

    #[test]
    fn reload_restarts_and_vanished_registrants_drop() {
        let mut tracker = DemandTracker::default();
        let start = Instant::now();
        tracker.observe(&[stats(0, "m", 5_000_000_000, 0), stats(1, "m", 0, 0)], start);
        // Counters went backwards on device 0 (engine reloaded).
        let demands = tracker.observe(&[stats(0, "m", 10, 0)], start + Duration::from_secs(1));
        assert_eq!(demands, vec![INITIAL_DEMAND]);
        assert_eq!(tracker.samples.len(), 1);
    }
}
