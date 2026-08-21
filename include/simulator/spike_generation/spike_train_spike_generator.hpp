#pragma once

#include "common/types.hpp"

#include <array>
#include <cstddef>

namespace nevresim {

/// Replay host-provided per-cycle input spikes (cycle-major layout).
///
/// ``input`` must expose ``data()`` and hold ``InputSize * SimulationLength``
/// values.  Index ``cycle * InputSize + neuron`` is the spike at ``cycle`` for
/// axon ``neuron``.  Cycles ``>= SimulationLength`` emit zero spikes (latency
/// padding handled by ``SpikingExecution`` loop length).
///
/// ``PreserveMultiplicity`` is the carry seam's alphabet. A per-cycle law's
/// producer emits at most one spike per neuron per cycle, so binarizing on load
/// is lossless and stays the default, byte for byte. A per-event producer emits
/// a COUNT, and the file carries it losslessly — so the counted seam passes the
/// value through, clamped nowhere: a ``!= 0`` here would silently deliver 1
/// where the producing segment fired 3, which is the exact multiplicity the
/// downstream fold is supposed to consume.
template <
    std::size_t InputSize,
    int SimulationLength,
    bool PreserveMultiplicity = false>
class SpikeTrainSpikeGeneratorImpl
{
public:
    using spike_source_t = std::array<spike_t, InputSize>;

    constexpr
    static spike_source_t generate_spikes(
        const auto& input, int cycle)
    {
        spike_source_t source{};
        if (cycle < 0 || cycle >= SimulationLength) {
            return source;
        }

        const auto* data = input.data();
        const std::size_t base =
            static_cast<std::size_t>(cycle) * InputSize;
        for (std::size_t neuron = 0; neuron < InputSize; ++neuron) {
            if constexpr (PreserveMultiplicity) {
                source[neuron] = static_cast<spike_t>(data[base + neuron]);
            } else {
                source[neuron] = static_cast<spike_t>(
                    data[base + neuron] != raw_input_t{0});
            }
        }
        return source;
    }
};

/// The binary carry seam (default point) — the name every emitted main.cpp has
/// always used, and the same type it has always named.
template <std::size_t InputSize, int SimulationLength>
using SpikeTrainSpikeGenerator =
    SpikeTrainSpikeGeneratorImpl<InputSize, SimulationLength, false>;

/// The counted carry seam: multiplicities cross the segment boundary intact.
template <std::size_t InputSize, int SimulationLength>
using CountedSpikeTrainSpikeGenerator =
    SpikeTrainSpikeGeneratorImpl<InputSize, SimulationLength, true>;

} // namespace nevresim
