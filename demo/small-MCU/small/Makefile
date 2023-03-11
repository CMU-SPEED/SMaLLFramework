all: autoencoder.x dscnn.x resnet.x 

autoencoder.x: driver.cpp
	g++ -std=c++17 -O3 -Iquantized/include -DQUANTIZED -DMODEL=autoencoder driver.cpp  -o autoencoder.x

dscnn.x: driver.cpp
	g++ -std=c++17 -O3 -Iquantized/include -DQUANTIZED -DMODEL=dscnn driver.cpp -o dscnn.x

resnet.x: driver.cpp
	g++ -std=c++17 -O3 -Iquantized/include -DQUANTIZED -DMODEL=resnet driver.cpp -o resnet.x
