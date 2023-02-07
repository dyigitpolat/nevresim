#pragma once
#include "common/types.hpp"

namespace nevresim
{
    
struct DefaultFirePolicy
{
    static constexpr spike_t 
    fire(const auto& threshold, auto& membrane_potential)
    {
        spike_t spike = threshold < membrane_potential;

        if(spike)
        {
            membrane_potential -= threshold;
        }
        
        return spike;
    }
};

} // namespace nevresim