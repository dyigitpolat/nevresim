#pragma once

#include "input_loader.hpp"
#include "spike_generator.hpp"

#include <algorithm>
#include <cstddef>

namespace nevresim {

template <
    int SimulationLength,
    template<std::size_t InputSize> class SpikeProvider>
class SpikingExecution
{
public:
    constexpr static auto execute(
        const auto& input, 
        auto& chip,
        const auto& compute_function,
        const auto& output_buffer_read_function)
    {
        std::array<Weight<weight_t>, chip.output_size_> buffer{};
        for(int i = 0; i < SimulationLength; ++i){
            chip.feed_input_buffer(
                SpikeProvider<chip.input_size_>::generate_spikes(input));

            compute_function(chip);

            auto&& out = output_buffer_read_function(chip);
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
    constexpr static auto execute(
        const auto& input, 
        auto& chip,
        const auto& compute_function,
        const auto& output_buffer_read_function)
    {
        for(std::size_t i = 0; i < chip.core_count_; ++i){
            chip.feed_real_input_buffer(input);
            compute_function(chip);
        }

        return output_buffer_read_function(chip);
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