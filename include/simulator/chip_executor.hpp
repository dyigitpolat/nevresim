#pragma once

#include "loaders/input_loader.hpp"
#include "simulator/spike_generation/spike_generator.hpp"
#include "simulator/compute_policy/real_valued_compute.hpp"
#include "simulator/compute_policy/spiking_compute.hpp"

#include <algorithm>
#include <cstddef>

namespace nevresim {

template <
    int SimulationLength,
    template<std::size_t InputSize> class SpikeProvider>
class SpikingExecution
{
public:
    template <typename C>
    using compute_policy_t = SpikingCompute<C>;

    constexpr static auto execute(
        const auto& input, 
        auto& chip,
        const auto& compute_function,
        const auto& output_buffer_read_function)
    {
        std::array<Weight<weight_t>, chip.config_.output_size_> buffer{};
        for(int i = 0; i < SimulationLength; ++i){
            const auto& spikes{
                SpikeProvider<chip.config_.input_size_>
                    ::generate_spikes(input)};

            compute_function(chip, spikes);

            auto&& out = output_buffer_read_function(chip, spikes);
            std::ranges::transform(
                std::ranges::cbegin(out), std::ranges::cend(out),
                std::ranges::cbegin(buffer), std::ranges::cend(buffer),
                std::ranges::begin(buffer), 
                std::plus{});
        }

        return buffer;
    }
};

class RealExecution
{
public:
    template <typename C>
    using compute_policy_t = RealValuedCompute<C>;

    constexpr static auto execute(
        const auto& input, 
        auto& chip,
        const auto& compute_function,
        const auto& output_buffer_read_function)
    {
        for(std::size_t i = 0; i < chip.config_.core_count_; ++i){
            compute_function(chip, input);
        }

        return output_buffer_read_function(chip, input);
    }
};

template <typename ExecutePolicy>
class ChipExecutor
{
public:
    constexpr static auto execute(
        const auto& input, 
        auto& chip,
        const auto& compute_function,
        const auto& output_buffer_read_function)
    {
        return ExecutePolicy::execute(
            input, chip, compute_function, output_buffer_read_function);
    }
};

} // namespace nevresim