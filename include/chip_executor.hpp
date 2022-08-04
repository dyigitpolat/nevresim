#pragma once

#include "input_loader.hpp"
#include "spike_generator.hpp"

#include <algorithm>

namespace nevresim {

template <int SimulationLength>
class SpikingExecution
{
public:
    constexpr static auto execute(
        const auto& input, 
        auto& chip,
        const auto& compute_function,
        const auto& output_buffer_read_function)
    {
        std::array<weight_t, chip.output_size_> buffer{};
        for(int i = 0; i < SimulationLength; ++i){
            chip.feed_input_buffer(
                SpikeGenerator<chip.input_size_>::generate_spikes(input));

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