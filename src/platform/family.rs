/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! Fleet vocabulary for devices: vendor and standardized GPU family.

use serde::Serialize;

#[derive(Serialize, Clone, Copy, PartialEq, Eq, Debug)]
#[serde(rename_all = "snake_case")]
#[allow(dead_code)]
pub(crate) enum GpuVendor {
    Nvidia,
    Apple,
    Amd,
    Unknown,
}

/// Standardized GPU family, the vocabulary shared with `muna deploy --gpu`
/// and the control plane's capacity records. Wire spelling is lowercase
/// (`"h100"`, `"b200"`). Mirrored by the control plane, which additionally
/// tolerates unknown slugs.
#[derive(Serialize, Clone, Copy, PartialEq, Eq, Debug)]
#[serde(rename_all = "lowercase")]
pub(crate) enum GpuFamily {
    Cpu,
    A10G,
    L4,
    L40S,
    A100,
    H100,
    H200,
    B200,
    B300,
    MI350X,
    MI355X,
}

impl GpuFamily {

    /// Derive the family from a device name as reported by NVML / ROCm
    /// (e.g. "NVIDIA H100 80GB HBM3" -> `H100`). Matches whole
    /// alphanumeric tokens, not substrings, so "NVIDIA L40S" never maps
    /// to `L4`. `None` for devices outside the fleet vocabulary; the
    /// control plane then falls back to raw-name matching.
    pub fn from_device_name(name: &str) -> Option<GpuFamily> {
        let tokens: Vec<String> = name
            .split(|c: char| !c.is_ascii_alphanumeric())
            .filter(|token| !token.is_empty())
            .map(str::to_uppercase)
            .collect();
        let has = |token: &str| tokens.iter().any(|t| t == token);
        [
            ("A10G",   GpuFamily::A10G),
            ("L40S",   GpuFamily::L40S),
            ("L4",     GpuFamily::L4),
            ("A100",   GpuFamily::A100),
            ("H100",   GpuFamily::H100),
            ("H200",   GpuFamily::H200),
            ("B200",   GpuFamily::B200),
            ("B300",   GpuFamily::B300),
            ("MI350X", GpuFamily::MI350X),
            ("MI355X", GpuFamily::MI355X),
        ]
        .into_iter()
        .find_map(|(token, family)| has(token).then_some(family))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn family_maps_real_device_names() {
        let cases = [
            ("NVIDIA H100 80GB HBM3",     Some(GpuFamily::H100)),
            ("NVIDIA H100 PCIe",          Some(GpuFamily::H100)),
            ("NVIDIA A100-SXM4-80GB",     Some(GpuFamily::A100)),
            ("NVIDIA A10G",               Some(GpuFamily::A10G)),
            ("NVIDIA B200",               Some(GpuFamily::B200)),
            ("AMD Instinct MI355X",       Some(GpuFamily::MI355X)),
            ("Apple M4 Pro",              None),
            ("Tesla T4",                  None),
        ];
        for (name, expected) in cases {
            assert_eq!(GpuFamily::from_device_name(name), expected, "{name}");
        }
    }

    /// "L4" is a substring of "L40S"; token matching keeps them distinct.
    #[test]
    fn family_does_not_confuse_l4_with_l40s() {
        assert_eq!(GpuFamily::from_device_name("NVIDIA L4"), Some(GpuFamily::L4));
        assert_eq!(GpuFamily::from_device_name("NVIDIA L40S"), Some(GpuFamily::L40S));
    }

    /// The wire spelling is the lowercase slug shared with `--gpu`.
    #[test]
    fn family_wire_spelling_is_lowercase() {
        assert_eq!(serde_json::to_value(GpuFamily::H100).unwrap(), "h100");
        assert_eq!(serde_json::to_value(GpuFamily::MI355X).unwrap(), "mi355x");
    }
}
