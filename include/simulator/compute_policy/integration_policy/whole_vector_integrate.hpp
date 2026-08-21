#pragma once

// The DEFAULT integration policy: one whole-vector reduction per cycle, then
// exactly ONE threshold evaluation on the reduced contribution.
//
// This is the arithmetic nevresim has always executed, lifted verbatim out of
// NeuronCompute<Config, SpikingCompute<...>>::leaky_integrate so that the
// integration step becomes a declared axis instead of a hard-coded call. The
// expression below is preserved LINE FOR LINE — association
// ``((inner_product) + bias) - leak`` and the ``static_cast<Membrane>(0)``
// accumulator seed included — because for ``weight_t == double`` a different
// association is a different number, and every default-point run must stay
// bit-identical.

#include "common/types.hpp"

#include <cstddef>
#include <iterator>
#include <numeric>

namespace nevresim
{

struct WholeVectorIntegrate
{
    /// The membrane a per-cycle law holds: the unbounded accumulator in the
    /// weight's own type, exactly as before this axis existed.
    template <typename WeightType>
    using membrane_t = MembranePotential<WeightType>;

    /// Whether one cycle can emit more than one spike per neuron. It cannot
    /// here: the recorders and the wire keep their binary raster.
    static constexpr bool counted_wire = false;

    template <
        typename FirePolicy,
        typename Weights,
        typename Signals,
        typename BiasT,
        typename LeakT,
        typename ThresholdT,
        typename Membrane>
    static constexpr spike_t step(
        const Weights& weights,
        const Signals& incoming_spikes,
        const BiasT& bias,
        const LeakT& leak,
        const ThresholdT& threshold,
        Membrane& membrane_potential)
    {
        membrane_potential +=
            std::inner_product(
                std::cbegin(weights), std::cend(weights),
                std::cbegin(incoming_spikes),
                static_cast<Membrane>(0))
            + bias
            - leak;

        return FirePolicy::fire(threshold, membrane_potential);
    }
};

} // namespace nevresim
