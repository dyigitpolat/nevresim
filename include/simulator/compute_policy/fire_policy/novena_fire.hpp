#pragma once
#include "common/types.hpp"

namespace nevresim
{
    
struct NovenaFirePolicy
{
    static constexpr bool always_on_every_cycle = true;
    static constexpr bool latency_gated = true;

    static constexpr spike_t 
    fire(const auto& threshold, auto& membrane_potential)
    {
        spike_t spike = threshold < membrane_potential;

        if(spike)
        {
            membrane_potential = 0;
        }
        
        return spike;
    }
};

} // namespace nevresim