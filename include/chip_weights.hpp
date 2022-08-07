#pragma once

#include <array>

#include "constants.hpp"
#include "types.hpp"

namespace nevresim {

template <
    std::size_t AxonCount >
class NeuronWeights
{
public:
    Threshold<weight_t> threshold_;
    std::array<Weight<weight_t>, AxonCount> weights_;
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