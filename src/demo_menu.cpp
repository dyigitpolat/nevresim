#include "demo_menu.hpp"

#include <iostream>
#include <limits>

DemoMenu::DemoMenu(const std::string &title) : m_title{title}
{
}

void DemoMenu::add_menu_item(const Demo &demo)
{
    m_demos.push_back(demo);
}

void DemoMenu::show() const
{
    print_menu();

    int user_input{};
    do
    {
        handle_stdin_error();

        std::cout << "Select an option (1-" << m_demos.size() << "): ";
        std::cin >> user_input;
    } while (user_input < 1 || user_input > static_cast<int>(m_demos.size()));

    m_demos[user_input - 1].play();

    char user_continue{};
    do
    {
        handle_stdin_error();

        print_title();
        std::cout << "Continue? (y / n): ";
        std::cin >> user_continue;
    } while (user_continue != 'y' && user_continue != 'n');
    std::cout << std::endl;

    if (user_continue == 'n')
    {
        return;
    }
    else
    {
        show();
    }
}

void DemoMenu::handle_stdin_error() const
{
    if (std::cin.fail())
    {
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
}

void DemoMenu::print_menu() const
{
    print_title();
    int i{}; 
    for (const Demo &d : m_demos)
    {
        std::cout << ++i << ": " << d.get_title() << std::endl;
    }
}

void DemoMenu::print_title() const
{
    std::cout << m_title << "..." << std::endl;
}