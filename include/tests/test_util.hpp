#pragma once

#include "chip.hpp"
#include "types.hpp"
#include "constants.hpp"

#include <iostream>
#include <iterator>
#include <algorithm>
#include <limits>

namespace nevresim
{
namespace tests
{

constexpr nevresim::core_id_t in = nevresim::k_input_buffer_id;
constexpr nevresim::core_id_t off = nevresim::k_no_connection;

template <
    auto connections,
    auto outputs,
    std::size_t axon_count,
    std::size_t neuron_count,
    std::size_t core_count,
    std::size_t input_size,
    std::size_t output_size,
    nevresim::MembraneLeak<weight_t> leak,
    typename ExecutePolicy
>
consteval auto generate_test_chip()
{
    using Cfg = nevresim::ChipConfiguration<
        axon_count, core_count, output_size>;

    constexpr Cfg config{connections, outputs};
    
    using Chip = nevresim::Chip<
        axon_count, neuron_count,
        core_count, input_size,
        output_size, config, leak,
        ExecutePolicy>;

    constexpr Chip chip{};
    return chip;
}

void print_prediction_summary(const auto& buffer)
{
    for(int q{}; auto x : buffer)
    {
        std::cout << "[" << q++ << ": " << x << "] ";
    }
    std::cout << "\n";
}

int argmax(const auto& buffer)
{
    return std::distance(
        buffer.begin(),
        std::max_element(buffer.begin(), buffer.end()));
}

void report_and_advance(
    int guess, int target, int idx, int& correct, int& total)
{
    total++;
    if(guess==target) correct++;

    std::cout 
        << idx << " prediction: " << guess
        << "\n";

    std::cout 
        << idx << " target: " << target
        << "\n";

    std::cout << correct << "/" << total << "\n";

    std::cout << "\n";
}

template <typename FloatType>
constexpr bool is_almost_equal(
    FloatType value_1, FloatType value_2)
{
    FloatType absolute_difference{
        (value_1 > value_2) ?
        (value_1 - value_2) :
        (value_2 - value_1)
    };

    return absolute_difference < 
        (std::numeric_limits<FloatType>::epsilon() * 2);
}

} // namespace nevresim::tests
} // namespace nevresim