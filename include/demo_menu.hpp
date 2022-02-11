#pragma once

#include "demo.hpp"

#include <vector> 
#include <string> 

class DemoMenu
{
private:
    std::vector<Demo> m_demos;
    std::string m_title;

public:
    DemoMenu(const std::string &title);

    void add_menu_item(const Demo& demo);
    void show() const;

private:
    void handle_stdin_error() const;
    void print_menu() const;
    void print_title() const;
};