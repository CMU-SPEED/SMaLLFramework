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
