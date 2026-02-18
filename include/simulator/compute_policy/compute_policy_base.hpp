#pragma once

#include "simulator/chip_utilities.hpp"

#include "common/constants.hpp"

namespace nevresim 
{

template <
    typename Chip, 
    typename SignalType, 
    bool AlwaysOnEveryCycle = true,
    bool LatencyGated = true>
class ComputePolicyBase
{
private:
    static constexpr
    SignalType retrieve_signal(
        const Chip& chip, 
        SpikeSource source, 
        const auto& input_buffer,
        [[maybe_unused]] int cycle = 0) 
    {
        if(source.core_ == k_input_buffer_id)
        {
            return input_buffer[source.neuron_];
        }
        else if(source.core_ == k_no_connection)
        {
            return SignalType{};
        }
        else if(source.core_ == k_always_on)
        {
            if constexpr (AlwaysOnEveryCycle)
            {
                return SignalType{1};
            }
            else
            {
                return cycle == 0 ? SignalType{1} : SignalType{};
            }
        }

        return 
            chip.get_cores()[source.core_]
                .get_output()[source.neuron_];
    }

    static constexpr
    auto get_axon_input(
        const Chip& chip, 
        core_id_t core_id, 
        const auto& input_buffer,
        int cycle = 0)
    {
        std::array<SignalType, Chip::config_.axon_count_> signals{};
        for(std::size_t axon_idx{}; axon_idx < Chip::config_.axon_count_; ++axon_idx)
        {
            signals[axon_idx] = retrieve_signal(
                chip, 
                Chip::mapping_.core_sources_[core_id].sources_[axon_idx], 
                input_buffer,
                cycle);
        }
        return signals;
    }

public:
    static constexpr
    void compute(Chip& chip, const auto& input_buffer, int cycle)
    {
        std::array<
            std::array<SignalType, Chip::config_.axon_count_>, 
            Chip::config_.core_count_> axons{};

        for(int core_id{}; auto& axon : axons)
        {
            axon = get_axon_input(chip, core_id++, input_buffer, cycle);
        }

        for(int core_id{}; auto& core : chip.get_cores())
        {
            if constexpr (LatencyGated)
            {
                if(cycle >= core.get_latency())
                {
                    core.compute(axons[core_id]);
                }
            }
            else
            {
                // TTFS latch: all cores active at every cycle.
                core.compute(axons[core_id]);
            }

            ++core_id;
        }
    }

    static constexpr
    auto read_output_buffer(
        const Chip& chip, const auto& input_buffer, int cycle = 0)
    {
        std::array<SignalType, Chip::config_.output_size_> output{};
        for(std::size_t output_idx{}; output_idx < Chip::config_.output_size_; ++output_idx)
        {
            output[output_idx] = retrieve_signal(
                chip, 
                Chip::mapping_.output_sources_[output_idx], 
                input_buffer,
                cycle);
        }
        return output;
    }
};

} // namespace nevresim
