#pragma once

#include "simulator/compute_policy/real_valued_compute.hpp"

#include <cstddef>

namespace nevresim {

class RealValuedExecution
{
public:
    using compute_policy_t = RealValuedCompute;

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

} // namespace nevresim