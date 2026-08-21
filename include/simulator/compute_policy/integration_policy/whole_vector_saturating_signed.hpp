#pragma once

// The SYNC-FIRE integration policy: one whole-vector reduction per cycle and
// exactly ONE threshold evaluation -- the same arithmetic
// `WholeVectorIntegrate` performs -- on a fixed-width TWO'S-COMPLEMENT
// membrane register that clamps to [-2^(Bits-1), 2^(Bits-1)-1].
//
// The reduction expression is preserved LINE FOR LINE from
// `whole_vector_integrate.hpp` -- association ((inner_product) + bias) - leak
// and the `static_cast<Membrane>(0)` accumulator seed included -- because for
// `weight_t == double` a different association is a different number, and the
// whole point of this policy is that it holds EXACTLY the number the unbounded
// accumulator holds.
//
// That equality is true only while neither rail is reached, so a rail is not
// this law's physics but a FAILURE: the deployment carries a static
// no-saturation bound proven over the mapped weights at generation time, and
// this policy asserts the same claim at run time. It throws rather than
// clamping silently, exactly as the count-currency ceiling does -- a clamp
// would report a number the deployment's own contract says it cannot produce.
//
// This is the same law the torch per-cycle kernels execute under
// `SomaLaw.membrane_bounds` (`models/spiking/lif_core_step.py`) and the
// generated sync-fire RTL executes in `hw/gen/odin_gen_core.v.tmpl`; the three
// must agree bit-for-bit.

#include "common/types.hpp"

#include <cstddef>
#include <iterator>
#include <numeric>
#include <stdexcept>

namespace nevresim
{

template <unsigned MembraneBits>
struct WholeVectorSaturatingSigned
{
    static_assert(
        MembraneBits >= 2 && MembraneBits <= 30,
        "membrane_bits must name a real two's-complement register width that "
        "fits every membrane type the chip is generated with");

    static constexpr unsigned membrane_bits = MembraneBits;

    /// The two's-complement register's value range.
    static constexpr long long membrane_floor =
        -(1LL << (MembraneBits - 1));
    static constexpr long long membrane_ceiling =
        (1LL << (MembraneBits - 1)) - 1LL;

    /// The membrane this law holds: the weight's own type, clamped into the
    /// register's interval, so the comparator keeps comparing `threshold_t`
    /// against the same type it always did.
    template <typename WeightType>
    using membrane_t = MembranePotential<WeightType>;

    /// One compare per cycle: at most one spike per neuron, so the wire and
    /// the recorders keep their binary raster.
    static constexpr bool counted_wire = false;

    template <typename Membrane>
    static constexpr bool touches_rail(Membrane value)
    {
        return value <= static_cast<Membrane>(membrane_floor)
            || value >= static_cast<Membrane>(membrane_ceiling);
    }

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

        if (touches_rail(membrane_potential))
        {
            throw std::logic_error(
                "WholeVectorSaturatingSigned: the membrane reached a rail of "
                "its two's-complement register. This law claims to hold the "
                "same number the unbounded accumulator holds, which is true "
                "only strictly inside the rails, and the deployment's "
                "no-saturation bound said it could not happen -- refused, "
                "never clamped, because a clamp reports a number the "
                "deployment's own contract denies");
        }

        return FirePolicy::fire(threshold, membrane_potential);
    }
};

} // namespace nevresim
