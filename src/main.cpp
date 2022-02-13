#include "demo_menu.hpp"

#include <iostream> 

//
#include <array>
#include <algorithm>
#include <numeric>
#include <cstdint>

namespace nevresim
{

using weight_t = int;
using threshold_t = weight_t;
using spike_t = int_fast8_t;

template <
    size_t AxonCount,
    threshold_t Threshold>
class Neuron
{
    using weights_array_t = std::array< weight_t, AxonCount>;

    weights_array_t weights_{};

public:
    constexpr 
    Neuron(weights_array_t weights) : weights_{weights} {}

    constexpr
    spike_t compute(const auto& axons) const
    {
        return Threshold < std::inner_product(
            std::begin(weights_), std::end(weights_),
            std::begin(axons), 0
        );
    }
};

template <
    size_t AxonCount,
    size_t NeuronCount,
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
    void compute(const auto& axons)
    {
        std::ranges::transform(
            neurons_, std::ranges::begin(output_spikes_),
            [&](const auto& neuron) { return neuron.compute(axons); }
        );
    }
};

template<
    size_t AxonCount,
    size_t NeuronCount,
    size_t CoreCount,
    threshold_t Threshold>
class Chip
{
    using neurons_array_t = 
        std::array< Core< AxonCount, NeuronCount, Threshold>, CoreCount>;

    neurons_array_t cores_{};

public:
    constexpr
    void compute(const auto& input_buffer)
    {
        
    }
};

} // namespace nevresim


void hello_world()
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

int main()
{
    //----------------------------------------------------//
    std::cout << "- - - - - - - - - - - - - -\n";
    std::cout << "- - - - - - - - - - - - - -\n";
    std::cout << "- -   N E V R E S I M   - -\n";
    std::cout << "- - - - - - - - - - - - - -\n";
    std::cout << "- - - - - - - - - - - - - -\n";
    //----------------------------------------------------//

    DemoMenu main_menu("MAIN MENU");
    main_menu.add_menu_item({hello_world, "Hello World!"});
    main_menu.show();

    return 0;
}