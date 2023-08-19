rm -rf *.so
echo "g++ -O3 -fopenmp -fPIC -I./include -I../include/small -I../include/small/platforms/reference --shared Conv2D.cpp -o libsmall.so"
g++ -O3 -fPIC -I./include -I../include -I../include/small -I../include/small/platforms/reference --shared Conv2D.cpp -o libsmall.so