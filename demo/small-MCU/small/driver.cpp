#include<stdio.h>


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
#endif
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