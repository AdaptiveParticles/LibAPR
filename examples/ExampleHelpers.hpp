//
// Created by bevan on 29/11/2020.
//

#ifndef LIBAPR_EXAMPLEHELPERS_H
#define LIBAPR_EXAMPLEHELPERS_H

#include <iostream>

bool command_option_exists(char **begin, char **end, const std::string &option)
{
    return std::find(begin, end, option) != end;
}

char* get_command_option(char **begin, char **end, const std::string &option)
{
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}




#endif //LIBAPR_EXAMPLEHELPERS_H
