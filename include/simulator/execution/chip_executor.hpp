#pragma once

namespace nevresim {

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