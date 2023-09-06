#!/bin/bash -x
rm -rf *.so
# g++ -O3 -fPIC -DZEN2 -DUARCH_ZEN2 -mavx2 -mfma -march=native -I./include -I../include -I../include/small -I../include/small/platforms/zen2 --shared layers.cpp -o libsmall.so
g++ -O3 -fPIC -DREF -DUARCH_REF -I./include -I../include -I../include/small -I../include/small/platforms/reference --shared layers.cpp -o libsmall.so