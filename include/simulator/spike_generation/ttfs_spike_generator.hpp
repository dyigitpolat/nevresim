#pragma once

#include "common/types.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace nevresim {

/// TTFS (Time-To-First-Spike) spike generator — latched mode.
///
/// Each input fires once and **stays high** for all subsequent cycles,
/// consistent with the TTFS latch semantics used in core outputs.
///
///     spike_time = round(SimulationLength * (1 - value))
///
/// - value = 1.0  →  latched high from cycle 0
/// - value > 0.0  →  latched high from cycle spike_time
/// - value = 0.0  →  spike_time = SimulationLength → never fires
///
/// The latched signal lets downstream neurons continuously accumulate
/// the input weight, so the membrane grows ∝ w_j · (t − t_j), tracking
/// the analytical TTFS model's w_j · a_j.
///
template <std::size_t InputSize, int SimulationLength>
class TTFSSpikeGenerator
{
public:
    using spike_source_t = std::array<spike_t, InputSize>;

    constexpr
    static spike_source_t generate_spikes(
        const auto& input, int cycle)
    {
        spike_source_t source{};
        std::transform(
            std::cbegin(input), std::cend(input),
            std::begin(source),
            [cycle](auto item)
            {
                // Clamp item to [0, 1]
                auto clamped = item;
                if (clamped < 0) clamped = 0;
                if (clamped > 1) clamped = 1;

                int spike_time = static_cast<int>(
                    std::llround(SimulationLength * (1.0 - clamped)));

                // Latched: once fired (cycle >= spike_time), stays high.
                // spike_time == SimulationLength means activation ≈ 0 → never fires.
                return static_cast<spike_t>(
                    spike_time < SimulationLength && cycle >= spike_time);
            });

        return source;
    }
};

} // namespace nevresim

