#pragma once

#include "weights_loader.hpp"
#include "constants.hpp"
#include "types.hpp"
#include "core.hpp"
#include "chip_utilities.hpp"
#include "spiking_compute.hpp"

#include <array>
#include <utility>
#include <cstddef>
#include <istream>

namespace nevresim {

template<
    std::size_t AxonCount,
    std::size_t NeuronCount,
    std::size_t CoreCount,
    std::size_t InputSize,
    std::size_t OutputSize,
    ChipConfiguration<
        AxonCount,
        CoreCount,
        OutputSize> Configuration,
    MembraneLeak<weight_t> LeakAmount,
    template <typename> typename ComputePolicy = SpikingCompute
    >
class Chip
{
    using cores_array_t = 
        std::array< 
            Core< AxonCount, NeuronCount, LeakAmount>, CoreCount>;

    using real_input_buffer_t = 
        std::array< raw_input_t, InputSize>;

    
    using input_buffer_t = 
        std::array<spike_t, InputSize>;

    cores_array_t cores_{};
    input_buffer_t input_buffer_{};
    real_input_buffer_t real_input_buffer_{};

public:
    static constexpr std::size_t core_count_{CoreCount};
    static constexpr std::size_t neuron_count_{NeuronCount};
    static constexpr std::size_t axon_count_{AxonCount};

    static constexpr std::size_t input_size_{InputSize};
    static constexpr std::size_t output_size_{OutputSize};

    static constexpr ChipConfiguration config_ = Configuration;

public:
    constexpr 
    Chip(cores_array_t cores) : cores_(cores) {}

    constexpr 
    Chip() : cores_{} {}

    constexpr 
    static auto generate_read_output_buffer()
    {
        return ComputePolicy<Chip>::generate_read_output_buffer();
    }

    consteval 
    static auto generate_compute() 
    {
        return ComputePolicy<Chip>::generate_compute();
    }

    constexpr
    void feed_input_buffer(const input_buffer_t& feed)
    {
        input_buffer_ = feed;
    }

    constexpr
    void feed_real_input_buffer(const auto& feed)
    {
        std::ranges::copy(feed, std::begin(real_input_buffer_)); 
    }

    constexpr
    void reset()
    {
        for(auto& core : cores_)
        {
            core.reset();
        }
    }

    constexpr
    void load_weights(
        const ChipWeights<CoreCount, NeuronCount, AxonCount>& weights)
    {
        for(auto core_iter = std::begin(weights.cores_); auto& core : cores_)
        {
            core.load_weights(*core_iter++);
        }
    }

    constexpr
    const cores_array_t& get_cores() const { return cores_; } 

    constexpr
    cores_array_t& get_cores() { return cores_; } 


    constexpr
    const input_buffer_t& get_input_buffer() const { return input_buffer_; } 

    constexpr
    input_buffer_t& get_input_buffer() { return input_buffer_; } 

    constexpr
    const real_input_buffer_t& get_real_input_buffer() const { return real_input_buffer_; } 

    constexpr
    real_input_buffer_t& get_real_input_buffer() { return real_input_buffer_; } 

};

} // namespace nevresim