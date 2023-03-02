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

#include "mbed.h"
#define NANO33BLE 0

// #include "unquantized/autoencoder.cpp"
// #include "unquantized/dscnn.cpp"
// #include "unquantized/resnet.cpp"

// #include "quantized/autoencoder.cpp"
// #include "quantized/resnet.cpp"
#include "quantized/dscnn.cpp"


void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
}

void loop() {
  // put your main code here, to run repeatedly:
  inference();
}
