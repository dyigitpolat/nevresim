#pragma once

// Constexpr self-tests for the event-serial integration policy: the fold IS
// the law, so it is checked at compile time in every generated binary exactly
// like the single-core and multi-core chips beside it.

#include "common/types.hpp"
#include "simulator/compute_policy/compare_policy/inclusive_compare.hpp"
#include "simulator/compute_policy/fire_policy/lif_fire_policy.hpp"
#include "simulator/compute_policy/integration_policy/event_serial_integrate.hpp"
#include "simulator/compute_policy/integration_policy/whole_vector_integrate.hpp"
#include "simulator/compute_policy/reset_policy/zero_reset.hpp"

#include <array>
#include <utility>

namespace nevresim
{
namespace tests
{

using SerialFire = LIFirePolicy<ZeroReset, InclusiveCompare>;

template <unsigned Bits, std::size_t A>
constexpr std::pair<spike_t, int> serial_fold(
    std::array<int, A> weights, std::array<spike_t, A> events, int threshold)
{
    int membrane{};
    const spike_t emitted =
        EventSerialIntegrate<Bits>::template step<SerialFire>(
            weights, events, 0, 0, threshold, membrane);
    return {emitted, membrane};
}

/// One spike per crossing, and the membrane hard-zeroes each time: three
/// occurrences of a supra-threshold slot are three spikes, not one.
constexpr bool test_event_serial_counts_every_crossing()
{
    const auto [emitted, membrane] =
        serial_fold<8>(std::array<int, 2>{5, 0},
                       std::array<spike_t, 2>{3, 0}, 4);
    return emitted == 3 && membrane == 0;
}

/// Ascending slot order with each slot's multiplicity ADJACENT: the second
/// occurrence of slot 0 crosses before slot 1's inhibition arrives.
///
/// SIGN-ASYMMETRIC on purpose (plan sec.2.3): +3 +3 -3 reaches 6 >= 5 and
/// emits 1, while the round-robin order +3 -3 +3 never leaves 3 and emits 0.
/// A symmetric witness (e.g. w={3,-1}) answers 1 under BOTH orders, so it
/// cannot separate them and an interleaving executor would compile clean.
constexpr bool test_event_serial_order_is_ascending_and_adjacent()
{
    const auto [emitted, membrane] =
        serial_fold<8>(std::array<int, 2>{3, -3},
                       std::array<spike_t, 2>{2, 1}, 5);
    return emitted == 1 && membrane == 0;
}

/// The membrane is an unsigned register: it floors at 0 instead of going
/// negative, and it ceilings at 2^bits - 1 instead of accumulating.
constexpr bool test_event_serial_membrane_saturates_both_rails()
{
    const auto [low_emitted, low_membrane] =
        serial_fold<8>(std::array<int, 1>{-3},
                       std::array<spike_t, 1>{4}, 100);
    const auto [high_emitted, high_membrane] =
        serial_fold<3>(std::array<int, 1>{5},
                       std::array<spike_t, 1>{3}, 100);
    return low_emitted == 0 && low_membrane == 0
        && high_emitted == 0 && high_membrane == 7;
}

/// The COUNT CURRENCY is the chip's own declared width, not a universal 127.
/// A chip narrower than a byte still gets the byte's currency, and no chip gets
/// more than one ``spike_t`` holds.
constexpr bool test_the_count_currency_is_the_chips_own_width()
{
    return EventSerialIntegrate<8>::emitted_spike_ceiling == 127
        && EventSerialIntegrate<3>::emitted_spike_ceiling == 127
        && EventSerialIntegrate<16>::emitted_spike_ceiling == 32767
        && EventSerialIntegrate<30>::emitted_spike_ceiling == 32767;
}

/// A 16-bit chip carries a count the 8-bit currency refuses: 200 occurrences of
/// a slot at theta emit 200 spikes in ONE cycle, unclamped. The same fold under
/// ``EventSerialIntegrate<8>`` would refuse this at 127.
constexpr bool test_event_serial_carries_a_count_over_the_stock_currency()
{
    const auto [emitted, membrane] =
        serial_fold<16>(std::array<int, 1>{100},
                        std::array<spike_t, 1>{200}, 100);
    return emitted == 200 && membrane == 0
        && emitted > EventSerialIntegrate<8>::emitted_spike_ceiling;
}

/// The default policy is the degenerate case: one reduction, one compare, at
/// most one spike — the arithmetic every pre-axes program executed.
constexpr bool test_whole_vector_integrate_fires_once_per_cycle()
{
    const std::array<int, 2> weights{5, 5};
    const std::array<spike_t, 2> events{3, 3};
    int membrane{};
    const spike_t emitted =
        WholeVectorIntegrate::step<SerialFire>(
            weights, events, 0, 0, 4, membrane);
    return emitted == 1 && membrane == 0;
}

constexpr bool test_event_serial_integrate()
{
    return test_event_serial_counts_every_crossing()
        && test_event_serial_order_is_ascending_and_adjacent()
        && test_event_serial_membrane_saturates_both_rails()
        && test_the_count_currency_is_the_chips_own_width()
        && test_event_serial_carries_a_count_over_the_stock_currency()
        && test_whole_vector_integrate_fires_once_per_cycle();
}

} // namespace nevresim::tests
} // namespace nevresim
