#include "mbed.h"

// #include "models/autoencoder.cpp"
// #include "models/dscnn.cpp"
// #include "models/resnet.cpp"

// #include "quant/autoencoder.cpp"
// #include "quant/resnet.cpp"
#include "quant/dscnn.cpp"

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
}

void loop() {
  // put your main code here, to run repeatedly:
  inference();
}
