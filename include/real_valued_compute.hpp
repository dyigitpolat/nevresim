#pragma once

#include "chip_utilities.hpp"

namespace nevresim 
{

template <typename Chip>
class RealValuedCompute
{
    consteval
    static auto retrieve_signal_from() 
    {
        return [](const Chip& chip, SpikeSource source) 
            -> MembranePotential<weight_t> 
            {
                if(source.core_ == k_input_buffer_id)
                {
                    return chip.get_real_input_buffer()[source.neuron_];
                }
                else if(source.core_ == k_no_connection)
                {
                    return MembranePotential<weight_t>{};
                }
                else
                {
                    return 
                        chip.get_cores()[source.core_]
                            .get_output_signals()[source.neuron_];
                }
            };
    }

    template <std::size_t N> consteval
    static auto retrieve_signals() 
    {
        return [](const Chip& chip, auto signal_sources) {
            std::array<MembranePotential<weight_t>, N> spikes{};

            [&]<std::size_t ...Idx> (std::index_sequence<Idx...>)
            {
                (
                    (spikes[Idx] 
                        = retrieve_signal_from()(chip, signal_sources[Idx])
                    )
                , ...);
            }(std::make_index_sequence<N>{});

            return spikes;
        };
    }

    consteval
    static auto get_input_for() 
    {
        return [](const Chip& chip, core_id_t core_id){
            return retrieve_signals<Chip::axon_count_>()(
                chip, Chip::config_.core_sources_[core_id].sources_);
        };
    }

public: 
    constexpr 
    static auto generate_read_output_buffer()
    {
        return [](const Chip& chip){
            return 
                retrieve_signals<Chip::output_size_>()(
                    chip, Chip::config_.output_sources_);
        };
    }

    consteval 
    static auto generate_compute() 
    {
        return [](Chip& chip){
            [] <core_id_t ...IDs>
            (Chip& chip_, std::index_sequence<IDs...>)
            {
                std::array<std::array<
                    MembranePotential<weight_t>, Chip::axon_count_>, 
                    Chip::core_count_> axons{};

                ((axons[IDs] = get_input_for()(chip_, IDs)), ...);
                (chip_.get_cores()[IDs].compute_real(axons[IDs]), ...);
                
            } (chip, std::make_index_sequence<Chip::core_count_>{});
        };
    }
};

} // namespace nevresim