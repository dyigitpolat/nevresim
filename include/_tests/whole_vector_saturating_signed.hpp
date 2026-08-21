#pragma once

// Constexpr self-tests for the sync-fire integration policy: the law claims to
// hold exactly the number the unbounded accumulator holds while the rails are
// unreached, so that claim is checked at compile time in every generated
// binary, beside the event-serial one.

#include "common/types.hpp"
#include "simulator/compute_policy/compare_policy/inclusive_compare.hpp"
#include "simulator/compute_policy/fire_policy/lif_fire_policy.hpp"
#include "simulator/compute_policy/integration_policy/whole_vector_integrate.hpp"
#include "simulator/compute_policy/integration_policy/whole_vector_saturating_signed.hpp"
#include "simulator/compute_policy/reset_policy/zero_reset.hpp"

#include <array>
#include <utility>

namespace nevresim
{
namespace tests
{

using SyncFire = LIFirePolicy<ZeroReset, InclusiveCompare>;

template <unsigned Bits, std::size_t A>
constexpr std::pair<spike_t, int> sync_fire_cycles(
    std::array<int, A> weights, std::array<std::array<spike_t, A>, 2> cycles,
    int threshold)
{
    int membrane{};
    spike_t emitted{};
    for (const auto& events : cycles)
    {
        emitted = static_cast<spike_t>(
            emitted
            + WholeVectorSaturatingSigned<Bits>::template step<SyncFire>(
                weights, events, 0, 0, threshold, membrane));
    }
    return {emitted, membrane};
}

template <std::size_t A>
constexpr std::pair<spike_t, int> unbounded_cycles(
    std::array<int, A> weights, std::array<std::array<spike_t, A>, 2> cycles,
    int threshold)
{
    int membrane{};
    spike_t emitted{};
    for (const auto& events : cycles)
    {
        emitted = static_cast<spike_t>(
            emitted
            + WholeVectorIntegrate::step<SyncFire>(
                weights, events, 0, 0, threshold, membrane));
    }
    return {emitted, membrane};
}

/// A NET-NEGATIVE cycle: the two-cycle window drives the membrane to -4 and
/// then back over threshold. An unsigned register would have floored the first
/// cycle at 0 and fired on the second (emitting 1 where the true law emits 0);
/// the two's-complement register carries the debt and the counts match the
/// unbounded accumulator exactly. This is the whole reason the value exists.
constexpr bool test_sync_fire_matches_the_unbounded_law_through_a_negative_cycle()
{
    const std::array<int, 2> weights{3, -7};
    const std::array<std::array<spike_t, 2>, 2> cycles{
        std::array<spike_t, 2>{1, 1}, std::array<spike_t, 2>{1, 0}};
    const auto [signed_emitted, signed_membrane] =
        sync_fire_cycles<16>(weights, cycles, 5);
    const auto [exact_emitted, exact_membrane] =
        unbounded_cycles(weights, cycles, 5);
    return signed_emitted == exact_emitted && signed_membrane == exact_membrane
        && signed_membrane < 0 && signed_emitted == 0;
}

/// At most one spike per neuron per cycle, exactly as the default policy: this
/// is a per-CYCLE law, and the wire stays binary.
constexpr bool test_sync_fire_fires_at_most_once_per_cycle()
{
    const std::array<int, 2> weights{5, 5};
    const std::array<std::array<spike_t, 2>, 2> cycles{
        std::array<spike_t, 2>{3, 3}, std::array<spike_t, 2>{0, 0}};
    const auto [emitted, membrane] = sync_fire_cycles<16>(weights, cycles, 4);
    return emitted == 1 && membrane == 0
        && !WholeVectorSaturatingSigned<16>::counted_wire;
}

/// The register interval is the two's-complement one, and the rail predicate
/// answers on its own bounds rather than on a hard-coded width.
constexpr bool test_sync_fire_declares_a_twos_complement_interval()
{
    return WholeVectorSaturatingSigned<16>::membrane_floor == -32768
        && WholeVectorSaturatingSigned<16>::membrane_ceiling == 32767
        && WholeVectorSaturatingSigned<8>::membrane_floor == -128
        && WholeVectorSaturatingSigned<8>::membrane_ceiling == 127
        && WholeVectorSaturatingSigned<8>::touches_rail(127)
        && WholeVectorSaturatingSigned<8>::touches_rail(-128)
        && !WholeVectorSaturatingSigned<8>::touches_rail(126);
}

constexpr bool test_whole_vector_saturating_signed()
{
    return test_sync_fire_matches_the_unbounded_law_through_a_negative_cycle()
        && test_sync_fire_fires_at_most_once_per_cycle()
        && test_sync_fire_declares_a_twos_complement_interval();
}

} // namespace nevresim::tests
} // namespace nevresim
