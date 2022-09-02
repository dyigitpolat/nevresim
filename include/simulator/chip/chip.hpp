#pragma once

#include "loaders/weights_loader.hpp"
#include "common/constants.hpp"
#include "common/types.hpp"
#include "simulator/chip/core.hpp"
#include "simulator/chip_utilities.hpp"
#include "simulator/chip_executor.hpp"

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

    cores_array_t cores_{};

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
};

} // namespace nevresim