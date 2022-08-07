#pragma once

#include <cstddef>

namespace nevresim {

using raw_input_t = double;

using weight_t = double;
using threshold_t = weight_t;
using membrane_potential_t = threshold_t;
using membrane_leak_t = membrane_potential_t;
using spike_t = std::int_fast8_t;

using core_id_t = std::size_t;
using axon_id_t = std::size_t;

template <typename WeightType>
struct ChipTypes
{
    using weight_t = WeightType;
    using threshold_t = weight_t;
    using membrane_potential_t = threshold_t;
    using membrane_leak_t = membrane_potential_t;
};

template <typename WeightType>
using chip_weight_t = 
    typename ChipTypes<WeightType>::weight_t;

template <typename WeightType>
using chip_threshold_t = 
    typename ChipTypes<WeightType>::threshold_t;

template <typename WeightType>
using chip_membrane_potential_t = 
    typename ChipTypes<WeightType>::membrane_potential_t;

template <typename WeightType>
using chip_membrane_leak_t = 
    typename ChipTypes<WeightType>::membrane_leak_t;

} // namespace nevresim