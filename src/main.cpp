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

struct SpikeSource
{
    core_id_t core_{};
    axon_id_t neuron_{};
};

template<std::size_t AxonCount>
struct CoreConnection {
    std::array<SpikeSource, AxonCount> sources_{};
};

constexpr core_id_t k_input_buffer_id{
    std::numeric_limits<core_id_t>::max()
};

template<
    std::size_t AxonCount,
    std::size_t CoreCount,
    std::size_t OutputSize>
struct ChipConfiguration {
    std::array<CoreConnection<AxonCount>, CoreCount> core_sources_{};
    std::array<SpikeSource, OutputSize> output_sources_{};
};

template <size_t N> consteval
auto make_spike_array(auto iter) 
{
    return 
        [&]<std::size_t ...Idx> (std::index_sequence<Idx...>)
        {
            return std::array<spike_t, N> { ((void)Idx, *(iter++))... };
        }(std::make_index_sequence<N>{});
} 

template<
    std::size_t AxonCount,
    std::size_t NeuronCount,
    std::size_t CoreCount,
    std::size_t InputSize,
    std::size_t OutputSize,
    threshold_t Threshold,
    ChipConfiguration<
        AxonCount,
        CoreCount,
        OutputSize> Configuration
    >
class Chip
{
    using cores_array_t = 
        std::array< Core< AxonCount, NeuronCount, Threshold>, CoreCount>;
    
    using input_buffer_t = 
        std::array< spike_t, InputSize>;

    cores_array_t cores_{};
    input_buffer_t input_buffer_{};

    consteval
    static auto retrieve_spike_from() 
    {
        return [](const Chip& chip, SpikeSource source) -> spike_t {
            if(source.core_ == k_input_buffer_id)
            {
                return chip.input_buffer_[source.neuron_];
            }
            else
            {
                return 
                    chip.cores_[source.core_]
                        .get_output_spikes()[source.neuron_];
            }
        };
    }

    template <std::size_t N> consteval
    static auto retrieve_spikes() 
    {
        /*
        return [](const Chip& chip, auto spike_sources) {
            constexpr auto spikes_view = 
                std::ranges::subrange(spike_sources) |
                std::views::transform(
                    [chip](auto source)
                    {  
                        return retrieve_spike_from()(chip, source);
                    });
            
            return make_spike_array<N>(std::ranges::begin(spikes_view));
        };
        */
        return [](const Chip& chip, auto spike_sources) {
            std::array<spike_t, N> spikes{};

            [&]<std::size_t ...Idx> (std::index_sequence<Idx...>)
            {
                (
                    (spikes[Idx] 
                        = retrieve_spike_from()(chip, spike_sources[Idx])
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
            return retrieve_spikes<AxonCount>()(
                chip, Configuration.core_sources_[core_id].sources_);
        };
    }

public:
    constexpr 
    Chip(cores_array_t cores) : cores_(cores) {}
    
    constexpr 
    static auto generate_read_output_buffer()
    {
        return [](const Chip& chip){
            return 
                retrieve_spikes<OutputSize>()(
                    chip, Configuration.output_sources_);
        };
    }

    consteval 
    static auto generate_compute() 
    {
        return [](Chip& chip){
            [] <core_id_t ...IDs>
            (Chip& chip_, std::index_sequence<IDs...>)
            {
                (chip_.cores_[IDs].compute(get_input_for()(chip_, IDs)), ...);
            } (chip, std::make_index_sequence<CoreCount>{});
        };
    }

    void feed_input_buffer(const input_buffer_t& feed)
    {
        input_buffer_ = feed;
    }
};

} // namespace nevresim


void test_single_core()
{
    nevresim::Core<3,3,10> core{
        std::array{
            nevresim::Neuron<3,10>{std::array{1,2,3}},
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


void test_single_core_chip()
{
    constexpr nevresim::ChipConfiguration<3,1,3> config {
        {nevresim::CoreConnection<3>{
            std::array<nevresim::SpikeSource, 3>{
                nevresim::SpikeSource{nevresim::k_input_buffer_id, 0},
                nevresim::SpikeSource{nevresim::k_input_buffer_id, 1},
                nevresim::SpikeSource{nevresim::k_input_buffer_id, 2}}
        }},
        {
            std::array<nevresim::SpikeSource, 3>{
                nevresim::SpikeSource{0, 0},
                nevresim::SpikeSource{0, 1},
                nevresim::SpikeSource{0, 2}}        
        }
    };

    nevresim::Chip<3,3,1,3,3,10, config> chip {
        std::array<nevresim::Core<3,3,10>, 1> {
            nevresim::Core<3,3,10> {
                std::array{
                    nevresim::Neuron<3,10>{std::array{1,2,3}},
                    nevresim::Neuron<3,10>{std::array{4,5,6}},
                    nevresim::Neuron<3,10>{std::array{7,8,9}}}
                }
        }
    };

    std::array<nevresim::spike_t, 3> input{1,1,0};
    chip.feed_input_buffer(input);
    nevresim::Chip<3,3,1,3,3,10, config>::generate_compute()(chip);
    for(auto spike : chip.generate_read_output_buffer()(chip))
    {
        std::cout << static_cast<int> (spike);
    }
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
    main_menu.add_menu_item({test_single_core, "test single core"});
    main_menu.add_menu_item({test_single_core_chip, "test single core chip"});
    main_menu.show();

    return 0;
}