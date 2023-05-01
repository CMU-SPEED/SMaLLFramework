//****************************************************************************
// SMaLL, Software for Machine Learning Libraries
// Copyright 2023 by The SMaLL Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM23-0126
//****************************************************************************

#include <stdio.h>

// #include "unquantized/autoencoder.cpp"
// #include "unquantized/dscnn.cpp"
// #include "unquantized/resnet.cpp"

#define autoencoder 0
#define dscnn 1
#define resnet 2

#ifndef MODEL
#define MODEL resnet
#endif

#if MODEL==autoencoder
#include "quantized/autoencoder.cpp"
#elif MODEL==resnet
#include "quantized/resnet.cpp"
#elif MODEL==dscnn
#include "quantized/dscnn.cpp"
#else
#error ERROR: Unrecognized MODEL macro.
#endif

//****************************************************************************
int main(int argc, char ** argv)
{
#if MODEL == autoencoder
    printf("quantized/autoencoder.cpp");
#elif MODEL == resnet
    printf("quantized/resnet.cpp");
#elif MODEL == dscnn
    printf("quantized/dscnn.cpp");
#endif
    inference();
}
