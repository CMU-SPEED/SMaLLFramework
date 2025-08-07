//****************************************************************************
// SMaLL, Software for Machine Learning Libraries
// Copyright 2025 by The SMaLL Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM23-0126
//****************************************************************************

#pragma once

#include <string>

#include <fstream>
#include <iostream>
#include <iomanip>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#include <regex>
#pragma GCC diagnostic pop

namespace small
{

//****************************************************************************
/****** Helper Functions ***** */
//****************************************************************************

//****************************************************************************
void extract_file_content(std::string const &filepath,
                          std::string       &contents)
{
    std::filebuf fb;

    if (fb.open(filepath, std::ios::in))
    {
        std::istream is(&fb);
        while (is)
        {
            contents += char(is.get());
        }
        fb.close();
    }
    else
    {
        std::cerr << "Failed to open " << filepath << std::endl;
    }
}

//****************************************************************************
template<typename BufferT>
BufferT extract_param(std::string const &state_dict_raw_data,
                      std::string        param_name)
{
    try
    {
        std::smatch start_match, end_match, element_match;
        int start = 0;
        int pos = -1;
        while ((pos = param_name.substr(start).find(".")) != -1)
        {
            param_name = param_name.insert(start+pos, "\\");
            start += pos+2;
        }
        std::regex start_regex("\"" + param_name + "\":\\s*\\[");
        std::regex end_regex("\\](\\,\\s*\"|\\})");
        std::regex element_regex("[\\-\\de\\.]+");
        std::regex_search(state_dict_raw_data, start_match, start_regex);
        std::string raw_data_substr =
            state_dict_raw_data.substr(start_match.position() +
                                       start_match.length());
        std::regex_search(raw_data_substr, end_match, end_regex);
        std::vector<typename BufferT::value_type> data;

        std::string search_str = raw_data_substr.substr(0, end_match.position());
        while (std::regex_search(search_str, element_match, element_regex))
        {
            data.push_back(std::stod(element_match[0]));
            search_str = search_str.substr(element_match.position() +
                                           element_match.length());
        }
        BufferT param_buf(data.size());
        // was std::move
        std::copy(data.begin(), data.end(), param_buf.data());
        //std::cout << param_name << " " << param_buf.size() << std::endl;
        return param_buf;
    }
    catch(std::exception &e)
    {
        std::cerr << e.what() << std::endl;
    }
    return BufferT();
}

}
