#include "demo_menu.hpp"

#include <iostream> 

#include "neuron.hpp"
#include "core.hpp"
#include "chip.hpp"

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
        std::cout << static_cast<int> (spike); //expected 001
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
    chip.generate_compute()(chip);
    for(auto spike : chip.generate_read_output_buffer()(chip))
    {
        std::cout << static_cast<int> (spike); //expected 001
    }
}


consteval auto generate_multi_core_chip()
{
    constexpr std::size_t axon_count{3};
    constexpr std::size_t neuron_count{3};
    constexpr std::size_t core_count{2};
    constexpr std::size_t input_size{3};
    constexpr std::size_t output_size{3};
    constexpr nevresim::threshold_t threshold{2};
    constexpr nevresim::membrane_leak_t leak{0};

    using Cfg = nevresim::ChipConfiguration<
        axon_count, core_count, output_size>;
    using Src = nevresim::SpikeSource;
    using Con = nevresim::CoreConnection<axon_count>;
    using Neu = nevresim::Neuron<
        axon_count, threshold, leak>;
    using Core = nevresim::Core<
        axon_count, neuron_count, threshold>;

    constexpr nevresim::core_id_t in = nevresim::k_input_buffer_id;

    constexpr Cfg config{{
            //core connectivity info
            Con{{ Src{in,0}, Src{in,1}, Src{in,2} }},
            Con{{ Src{0,2},  Src{0,0},  Src{0,1}  }}},
            
            //output buffer description
            { Src{1,2}, Src{1,1}, Src{1,0} }
        };

        
    using Chip = nevresim::Chip<
        axon_count, neuron_count, 
        core_count, input_size, 
        output_size, threshold, config>;

    constexpr Chip chip{{
            // weights
            Core{{ Neu{{0,0,1}}, Neu{{0,1,1}}, Neu{{1,1,1}} }},
            Core{{ Neu{{0,1,0}}, Neu{{1,0,0}}, Neu{{1,0,1}} }}
        }};
    
    return chip;
}

void test_multi_core_chip()
{
    auto chip = generate_multi_core_chip();

    std::array<nevresim::spike_t, 3> input{1,1,1};
    chip.feed_input_buffer(input);
    auto compute = chip.generate_compute();
    compute(chip);
    compute(chip);

    for(auto spike : chip.generate_read_output_buffer()(chip))
    {
        std::cout << static_cast<int> (spike); //expected 100
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
    main_menu.add_menu_item({test_multi_core_chip, "test multi core chip"});
    main_menu.show();

    return 0;
}