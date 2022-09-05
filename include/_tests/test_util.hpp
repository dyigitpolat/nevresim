#pragma once

#include "simulator/chip/chip.hpp"
#include "common/types.hpp"
#include "common/constants.hpp"

#include <iostream>
#include <iterator>
#include <algorithm>
#include <limits>
#include <fstream>

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
    template <typename> typename ConcreteComputePolicy
>
consteval auto generate_test_chip()
{

    using Cfg = nevresim::ChipConfiguration<
        axon_count,
        neuron_count,
        core_count,
        input_size,
        output_size,
        leak
    >;

    using Map = Mapping<Cfg>;

    constexpr Map mapping{connections, outputs};
    
    using Chip = Chip<Cfg, mapping, ConcreteComputePolicy>;

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
        << idx << ") prediction: " << guess
        << "\n";

    std::cout 
        << idx << ") target: " << target
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

void load_weights(auto& chip, auto weights_filename)
{
    WeightsLoader<
        chip.config_> 
        weights_loader{};

    std::ifstream weights_stream(weights_filename);
    if(weights_stream.is_open())
    {
        weights_stream >> weights_loader;
    }

    chip.load_weights(weights_loader.chip_weights_);
}

auto load_input_n(auto input_filename_prefix, int input_id)
{
    InputLoader input_loader{};
    
    std::string fname = 
        std::string{input_filename_prefix} +
        std::to_string(input_id) +
        std::string{".txt"};

    std::ifstream input_stream(fname);
    if(input_stream.is_open())
    {
        input_stream >> input_loader;
    }

    return std::pair{input_loader.input_, input_loader.target_};
}

void test_on_inputs(
    auto& chip, 
    auto input_filename_prefix, 
    int input_count, 
    auto executor)
{
    int correct{};
    int total{};
    for(int idx = 0; idx < input_count; ++idx)
    {
        auto [input, target] = load_input_n(input_filename_prefix, idx);
        auto buffer = chip.execute(input, executor);

        chip.reset();

        print_prediction_summary(buffer);
        int guess = argmax(buffer);
        report_and_advance(
            guess, target, idx, correct, total);
    }
}

} // namespace nevresim::tests
} // namespace nevresim