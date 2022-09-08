#pragma once

#include "common/types.hpp"

#include <array>

namespace nevresim
{

template <
    std::size_t AxonCount,
    std::size_t NeuronCount,
    std::size_t CoreCount,
    std::size_t InputSize,
    std::size_t OutputSize,
    MembraneLeak<weight_t> LeakAmount
    >
struct ChipConfiguration
{
    static constexpr std::size_t core_count_{CoreCount};
    static constexpr std::size_t neuron_count_{NeuronCount};
    static constexpr std::size_t axon_count_{AxonCount};
    static constexpr std::size_t input_size_{InputSize};
    static constexpr std::size_t output_size_{OutputSize};

    static constexpr MembraneLeak<weight_t> leak_amount_{LeakAmount};
};

struct SpikeSource
{
    core_id_t core_{};
    axon_id_t neuron_{};
};

template<std::size_t AxonCount>
struct CoreConnection {
    std::array<SpikeSource, AxonCount> sources_{};
};

template<typename Config>
struct Mapping {
    std::array<CoreConnection<Config::axon_count_>, Config::core_count_> 
        core_sources_{};
        
    std::array<SpikeSource, Config::output_size_> output_sources_{};
};

} // namespace nevresim