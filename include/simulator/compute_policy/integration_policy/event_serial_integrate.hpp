#pragma once

// The EVENT-SERIAL integration policy: the threshold is evaluated after every
// arriving event OCCURRENCE, in the canonical order, and a neuron may emit more
// than one spike in a single cycle.
//
//     for axon a = 0 .. A-1:                 // ascending slot order
//         repeat e[a] times:                 // occurrences of one slot ADJACENT
//             m := sat(m + w[a])             // clamp to [0, 2^Bits - 1]
//             if compare(theta, m): emit one spike; reset(m)
//     return the per-neuron COUNT emitted this cycle
//
// This is the same fold the torch twins execute
// (``mimarsinan/models/spiking/serial/fold.py``); the two implementations must
// agree bit-for-bit, so every ordering decision here is the one written down
// there: ascending axon index, occurrences of one slot adjacent, saturate
// BEFORE the compare, hard-zero reset per crossing.
//
// Preconditions this policy REFUSES to run without, rather than approximate:
//   * the per-neuron ``bias_`` and the per-cycle ``leak`` are zero. Under this
//     law the bias is delivered by the mapper as an always-on axon ROW at the
//     tail of the slot order (the param-encoded bias mode the point requires),
//     so a nonzero per-cycle scalar would be charge with no position in the
//     event order — and a leak is a per-cycle mechanism the per-event law has
//     no plumbing for.
//   * event multiplicities are non-negative. A negative arrival is a sign
//     folded into the wire instead of the weight.
//   * the emitted count fits the count currency (127). It is refused, NEVER
//     wrapped: ``spike_t`` is ``int_fast8_t`` and a silent saturation there is
//     exactly the failure the count currency exists to make impossible.

#include "common/types.hpp"

#include <cstddef>
#include <stdexcept>

namespace nevresim
{

/// The count-currency ceiling: one window/cycle count travels as one
/// ``spike_t``, and every implementation of the serial fold shares this bound.
inline constexpr int k_emitted_spike_ceiling = 127;

template <unsigned MembraneBits>
struct EventSerialIntegrate
{
    static_assert(
        MembraneBits >= 1 && MembraneBits <= 30,
        "membrane_bits must name a real register width that fits every "
        "membrane type the chip is generated with");

    static constexpr unsigned membrane_bits = MembraneBits;

    /// The saturating unsigned register's value range. The C++ storage type
    /// stays the weight's own type (see ``membrane_t``); the RANGE is what the
    /// register is, and every update clamps into it.
    static constexpr long long membrane_floor = 0;
    static constexpr long long membrane_ceiling =
        (1LL << MembraneBits) - 1LL;

    /// The membrane this law holds. Deliberately the weight's own type rather
    /// than an unsigned accumulator: the values are clamped into
    /// ``[0, 2^Bits-1]`` on EVERY update, so the arithmetic is that of the
    /// unsigned register exactly, while the comparator keeps comparing
    /// ``threshold_t`` against the same type it always did — an unsigned
    /// membrane against a signed threshold is a signed/unsigned comparison
    /// (``-Wsign-compare``, and worse, wrap-around instead of a clamp).
    template <typename WeightType>
    using membrane_t = MembranePotential<WeightType>;

    /// One cycle CAN emit more than one spike per neuron here, so the wire and
    /// the recorders must carry counts, not bits.
    static constexpr bool counted_wire = true;

    template <typename Membrane>
    static constexpr Membrane saturate(Membrane value)
    {
        const Membrane floor_value = static_cast<Membrane>(membrane_floor);
        const Membrane ceiling_value = static_cast<Membrane>(membrane_ceiling);
        if (value < floor_value)
        {
            return floor_value;
        }
        if (value > ceiling_value)
        {
            return ceiling_value;
        }
        return value;
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
        if (!(bias == BiasT{}))
        {
            throw std::logic_error(
                "EventSerialIntegrate: the per-neuron bias_ must be zero under "
                "the event-serial law — the bias is delivered as an always-on "
                "axon row at the tail of the slot order (param-encoded bias), "
                "and a per-cycle scalar has no position in the event order");
        }
        if (!(leak == LeakT{}))
        {
            throw std::logic_error(
                "EventSerialIntegrate: the membrane leak must be zero under "
                "the event-serial law — leak is a per-CYCLE mechanism and this "
                "law has no per-event leak defined");
        }

        int emitted = 0;
        for (std::size_t axon = 0; axon < weights.size(); ++axon)
        {
            const auto multiplicity = incoming_spikes[axon];
            if (multiplicity < 0)
            {
                throw std::logic_error(
                    "EventSerialIntegrate: a NEGATIVE event multiplicity "
                    "arrived on an axon — a sign belongs in the weight, never "
                    "on the wire");
            }
            for (auto occurrence = multiplicity; occurrence > 0; --occurrence)
            {
                membrane_potential = saturate(
                    static_cast<Membrane>(membrane_potential + weights[axon]));
                emitted += static_cast<int>(
                    FirePolicy::fire(threshold, membrane_potential));
            }
        }

        if (emitted > k_emitted_spike_ceiling)
        {
            throw std::logic_error(
                "EventSerialIntegrate: a neuron emitted more spikes in ONE "
                "cycle than the count currency carries (127) — refused, never "
                "clamped, because a clamp reports a number the deployment "
                "cannot produce; lower the fan-in or raise theta");
        }
        return static_cast<spike_t>(emitted);
    }
};

} // namespace nevresim
