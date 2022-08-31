#pragma once

#include "simulator/chip_utilities.hpp"
#include "common/constants.hpp"
#include "common/types.hpp"

#include <array>

namespace nevresim {

template <ChipConfiguration Config>
class NeuronWeights
{
public:
    std::array<Weight<weight_t>, Config.axon_count_> weights_;
    Threshold<weight_t> threshold_{};
    Bias<weight_t> bias_{};
};

template <ChipConfiguration Config>
class CoreWeights
{
public:
    std::array<NeuronWeights<Config>, Config.neuron_count_> neurons_;
};

template <ChipConfiguration Config>
class ChipWeights
{
public:
    std::array<CoreWeights<Config>, Config.core_count_> cores_;
};

} // namespace nevresim