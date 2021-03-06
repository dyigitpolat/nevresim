#pragma once

#include "input_loader.hpp"
#include "spike_generator.hpp"

#include <algorithm>

namespace nevresim {

class SpikingExecution
{
public:
    constexpr static auto execute(
        const InputLoader& loader, 
        auto& chip,
        const auto& compute_function,
        const auto& output_buffer_read_function)
    {
        std::array<weight_t, chip.output_size_> buffer{};
        for(auto i : std::views::iota(0, 2000)){
            (void) i;

            chip.feed_input_buffer(
                SpikeGenerator<chip.input_size_>::generate_spikes(loader));

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
        const InputLoader& loader, 
        auto& chip,
        const auto& compute_function,
        const auto& output_buffer_read_function)
    {
        return ExecutePolicy::execute(
            loader, chip, compute_function, output_buffer_read_function);
    }
};

} // namespace nevresim