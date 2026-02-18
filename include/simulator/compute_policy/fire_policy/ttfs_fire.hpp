#pragma once
#include "common/types.hpp"

#include <limits>

namespace nevresim
{

struct TTFSFirePolicy
{
    /// TTFS latch mode: always-on (bias) sources fire every cycle,
    /// consistent with latched inputs and latched core outputs.
    static constexpr bool always_on_every_cycle = true;
    /// TTFS: no latency gating — all cores active from cycle 0.
    static constexpr bool latency_gated = false;

    /// TTFS fire-once + latch semantics:
    ///   - Fire when membrane_potential >= threshold  (i.e.  threshold <= mp)
    ///   - After firing, set membrane to sentinel (-1000000) and latch
    ///     output HIGH (return 1 on every subsequent call).
    ///
    /// The latch ensures downstream neurons continuously accumulate w_j
    /// from this axon, so their membrane grows ∝ sum(w_j · (t − t_j)),
    /// tracking the analytical TTFS model's sum(w_j · a_j).
    static constexpr spike_t
    fire(const auto& threshold, auto& membrane_potential)
    {
        // Already fired → latched high.
        if(membrane_potential <= -999999)
            return 1;

        spike_t spike = (threshold <= membrane_potential) ? 1 : 0;

        if(spike)
        {
            membrane_potential = -1000000;
        }

        return spike;
    }
};

} // namespace nevresim

