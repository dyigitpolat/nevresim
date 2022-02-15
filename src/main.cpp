#include "demo_menu.hpp"

#include <iostream> 

//
#include <array>
#include <algorithm>
#include <ranges>
#include <numeric>
#include <limits>
#include <cstdint>
#include <cstddef>
#include <utility>

namespace nevresim
{

using weight_t = int;
using threshold_t = weight_t;
using membrane_potential_t = threshold_t;
using membrane_leak_t = membrane_potential_t;
using spike_t = std::int_fast8_t;

using core_id_t = std::size_t;
using axon_id_t = std::size_t;

template <
    std::size_t AxonCount,
    threshold_t Threshold,
    membrane_leak_t LeakAmount = 0>
class Neuron
{
    using weights_array_t = std::array< weight_t, AxonCount>;

    weights_array_t weights_{};
    membrane_potential_t membrane_potential_{};

    constexpr void leaky_integrate(const auto& incoming_spikes)
    {
        membrane_potential_ += std::inner_product(
            std::begin(weights_), std::end(weights_),
            std::begin(incoming_spikes), 0
        ) - LeakAmount;
    }

    constexpr spike_t fire()
    {
        spike_t spike = Threshold < membrane_potential_;

        if(spike)
        {
            membrane_potential_ -= Threshold;
        }
        
        return spike;
    }

public:
    constexpr 
    Neuron(weights_array_t weights) : weights_{weights} {}

    constexpr
    spike_t compute(const auto& incoming_spikes)
    {
        leaky_integrate(incoming_spikes);
        return fire();
    }
};

template <
    std::size_t AxonCount,
    std::size_t NeuronCount,
    threshold_t Threshold>
class Core
{
    using neuron_t = Neuron<AxonCount, Threshold>;
    using neurons_array_t = std::array<neuron_t, NeuronCount>;
    using output_array_t = std::array<spike_t, NeuronCount>;

    neurons_array_t neurons_{};
    output_array_t output_spikes_{};

public:    
    constexpr 
    Core(neurons_array_t neurons) : neurons_(neurons) {}

    constexpr
    const output_array_t& get_output_spikes() const & 
    {
        return output_spikes_;
    }

    constexpr
    void compute(const auto& incoming_spikes)
    {
        std::ranges::transform(
            neurons_, std::ranges::begin(output_spikes_),
            [&](auto& neuron) { return neuron.compute(incoming_spikes); }
        );
    }
};

struct AxonSource
{
    core_id_t core_;
    axon_id_t axon_;
};

template<std::size_t AxonCount>
struct CoreConnection {
    std::array<AxonSource, AxonCount> sources_{};
};

template<
    std::size_t CoreCount, 
    std::size_t AxonCount,
    std::size_t InputCoreCount,
    std::size_t OutputCoreCount>
struct ChipConfiguration {
    std::array<CoreConnection<AxonCount>, CoreCount> core_sources_{};
    std::array<core_id_t, InputCoreCount> input_cores_{};
    std::array<core_id_t, OutputCoreCount> output_cores_{};
};

template<
    std::size_t AxonCount,
    std::size_t NeuronCount,
    std::size_t CoreCount,
    threshold_t Threshold,
    ChipConfiguration Configuration>
class Chip
{
    using cores_array_t = 
        std::array< Core< AxonCount, NeuronCount, Threshold>, CoreCount>;

    cores_array_t cores_{};

    consteval
    auto get_input_for(core_id_t core_id)
    {
        auto axon_sources_view = 
            Configuration.core_sources_[core_id] |
            std::views::transform([this](auto source){
                return cores_[source.core_].get_output_spikes()[source.axon_];
            });
        
        return std::vector{
            std::ranges::begin(axon_sources_view), 
            std::ranges::end(axon_sources_view)
        };
    }

public:
    constexpr 
    Chip(cores_array_t cores) : cores_(cores) {}

    constexpr
    void compute(const auto& incoming_spikes)
    {
        [this] <core_id_t ...IDs>
        (std::index_sequence<IDs...>)
        {
            (cores_[IDs].compute(get_input_for(IDs)), ...);
        } (std::make_index_sequence<CoreCount>{});
    }
};

} // namespace nevresim


void test_simple_core()
{
    nevresim::Core<3,2,10> core{
        std::array{
            nevresim::Neuron<3,10>{std::array{4,5,6}},
            nevresim::Neuron<3,10>{std::array{7,8,9}}}
    };

    std::array<nevresim::spike_t, 3> input{1,1,0};

    core.compute(input);

    for(auto spike : core.get_output_spikes())
    {
        std::cout << static_cast<int> (spike);
    }
}

void test_simple_chip()
{
    
}

int main()
{
    //-----------------------------------------//
    std::cout << "- - - - - - - - - - - - - -\n";
    std::cout << "- - - - - - - - - - - - - -\n";
    std::cout << "- -   N E V R E S I M   - -\n";
    std::cout << "- - - - - - - - - - - - - -\n";
    std::cout << "- - - - - - - - - - - - - -\n";
    //-----------------------------------------//

    DemoMenu main_menu("MAIN MENU");
    main_menu.add_menu_item({test_simple_core, "test simple core"});
    main_menu.show();

    return 0;
}