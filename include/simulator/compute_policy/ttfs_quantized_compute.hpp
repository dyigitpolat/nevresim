#pragma once

#include "simulator/compute_policy/compute_policy_base.hpp"

namespace nevresim 
{

/// Compute policy for TTFS quantized (cycle-based) mode.
///
/// Each neuron internally simulates S discrete time steps per the
/// B1-model TTFS dynamics:
///
///   Phase 1:  V = W · a + b   (charge from incoming activations)
///   Phase 2:  for k = 0 … S-1:
///                 if V >= θ  →  output (S − k) / S  (fire-once)
///                 V += θ / S  (constant ramp)
///
/// From the chip's perspective the policy is identical to
/// ``TTFSAnalyticalCompute``: real-valued signals, bias always active,
/// no latency-gating (cores are processed in topological order via
/// repeated sweeps).
///
/// Template parameter ``S`` is the number of discrete time steps.
///
template <int S>
class TTFSQuantizedCompute
{
public:
    using signal_t = double;
    static constexpr int time_steps = S;

    template <typename Chip>
    using concrete_policy_t = ComputePolicyBase<Chip, signal_t, true, false>;
};

} // namespace nevresim

