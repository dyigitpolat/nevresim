#pragma once

#include "weights_loader.hpp"
#include "constants.hpp"
#include "types.hpp"
#include "core.hpp"
#include "chip_utilities.hpp"
#include "chip_executor.hpp"

#include <array>
#include <utility>
#include <cstddef>
#include <istream>

namespace nevresim {

template<
    typename Config,
    Mapping<Config> mapping,
    typename ExecutePolicy
    >
class Chip
{
    template <typename C>
    using compute_policy_t = typename ExecutePolicy::compute_policy_t<C>;

    using core_t = Core<Config, typename compute_policy_t<Chip>::base_t>;
    using cores_array_t = 
        std::array<core_t, Config::core_count_>;
    using input_buffer_t = 
        std::array<
            typename compute_policy_t<Chip>::signal_t, 
            Config::input_size_>;

    cores_array_t cores_{};
    input_buffer_t input_buffer_{};

public:
    static constexpr Config config_ = Config{};
    static constexpr Mapping mapping_ = mapping;

public:
    constexpr 
    Chip(cores_array_t cores) : cores_(cores) {}

    constexpr 
    Chip() : cores_{} {}
    
    constexpr 
    auto execute(const auto& input)
    {
        return ChipExecutor<ExecutePolicy>::execute(
            input, 
            *this, 
            compute_policy_t<Chip>::generate_compute(), 
            compute_policy_t<Chip>::generate_read_output_buffer()
        );
    }

    constexpr
    void feed_input_buffer(const auto& feed)
    {
        std::ranges::copy(feed, std::begin(input_buffer_)); 
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
        const ChipWeights<config_>& weights)
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
};

} // namespace nevresim