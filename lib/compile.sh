rm -rf *.so
echo "g++ -O3 -fPIC -I./include -I../include/small -I../include/small/platforms/reference --shared Conv2D.cpp -o libsmall_conv2d.so"
g++ -O3 -fPIC -I./include -I../include -I../include/small -I../include/small/platforms/reference --shared Conv2D.cpp -o libsmall_conv2d.so