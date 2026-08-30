#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

namespace nevresim {

template<typename T>
struct constant
{
    T value;
    consteval constant(T val) : value(val) {}
};

using raw_input_t = double;

/// The COUNT CURRENCY's word. A per-cycle spike count crosses every wire in
/// this simulator as one ``spike_t``, so its width is the widest count any
/// chip built here can carry — exactly 16 bits, on every platform, because a
/// deployment's counts must not depend on how a compiler sizes a "fast" type.
using spike_t = std::int16_t;

/// A chip narrower than a byte still gets the byte's currency: no
/// implementation of the fold carries counts in a sub-byte word.
inline constexpr unsigned k_count_currency_floor_bits = 8;

/// The count currency of ONE chip, from the chip's own declared arithmetic
/// width. CROSS-LANGUAGE CONTRACT with ``count_ceiling`` in
/// ``mimarsinan/models/spiking/serial/refusals.py``: a per-window count travels
/// in a SIGNED word of the width the chip declares its registers at, never
/// narrower than a byte and never wider than ``spike_t``.
constexpr int count_currency_ceiling(unsigned membrane_bits)
{
    const unsigned bits = membrane_bits < k_count_currency_floor_bits
        ? k_count_currency_floor_bits
        : membrane_bits;
    const long long claimed = (1LL << (bits - 1)) - 1LL;
    const long long carried =
        static_cast<long long>(std::numeric_limits<spike_t>::max());
    return static_cast<int>(claimed < carried ? claimed : carried);
}

using core_id_t = std::size_t;
using axon_id_t = std::size_t;

template <typename WeightType>
using Weight = WeightType;

template <typename WeightType>
using Threshold = WeightType;

template <typename WeightType>
using MembranePotential = WeightType;

template <typename WeightType>
using Bias = WeightType;

template <typename WeightType>
using MembraneLeak = WeightType;

} // namespace nevresim