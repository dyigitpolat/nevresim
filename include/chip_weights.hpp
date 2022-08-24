#pragma once

#include "constants.hpp"
#include "types.hpp"

#include <array>

namespace nevresim {

template <
    std::size_t AxonCount >
class NeuronWeights
{
public:
    std::array<Weight<weight_t>, AxonCount> weights_;
    Threshold<weight_t> threshold_{};
    Bias<weight_t> bias_{};
};

template <
    std::size_t NeuronCount,
    std::size_t AxonCount >
class CoreWeights
{
public:
    std::array<NeuronWeights<AxonCount>, NeuronCount> neurons_;
};

template <
    std::size_t CoreCount,
    std::size_t NeuronCount,
    std::size_t AxonCount >
class ChipWeights
{
public:
    std::array<CoreWeights<NeuronCount, AxonCount>, CoreCount> cores_;
};

} // namespace nevresim