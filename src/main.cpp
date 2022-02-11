#include "demo_menu.hpp"

#include <iostream> 

void hello_world()
{
    std::cout << "42." << std::endl;
}

int main()
{
    //----------------------------------------------------//
    std::cout << "- - - - - - - - - - - - - -" << std::endl;
    std::cout << "- - - - - - - - - - - - - -" << std::endl;
    std::cout << "-         Welcome!        -" << std::endl;
    std::cout << "  Demo Program Entry Point " << std::endl;
    std::cout << "- - - - - - - - - - - - - -" << std::endl;
    std::cout << "- - - - - - - - - - - - - -" << std::endl;
    //----------------------------------------------------//

    DemoMenu main_menu("MAIN MENU");
    main_menu.add_menu_item({hello_world, "Hello World!"});
    main_menu.show();

    return 0;
}