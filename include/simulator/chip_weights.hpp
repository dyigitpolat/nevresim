#pragma once

#include "simulator/chip_utilities.hpp"
#include "common/constants.hpp"
#include "common/types.hpp"

#include <array>

namespace nevresim {

template <ChipConfiguration Config, typename WeightType>
class NeuronWeights
{
    using weight_t = WeightType;

public:
    std::array<weight_t, Config.axon_count_> weights_;
    Threshold<weight_t> threshold_{};
    Bias<weight_t> bias_{};
};

template <ChipConfiguration Config, typename WeightType>
class CoreWeights
{
public:
    std::array<
        NeuronWeights<Config, WeightType>, 
        Config.neuron_count_> neurons_;
};

template <ChipConfiguration Config, typename WeightType>
class ChipWeights
{
public:
    std::array<CoreWeights<Config, WeightType>, Config.core_count_> cores_;
};

} // namespace nevresim