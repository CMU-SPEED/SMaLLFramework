all: autoencoder.x dscnn.x resnet.x 

autoencoder.x: driver.cpp
	g++ -std=c++17 -O3 -DMODEL=autoencoder driver.cpp  -o autoencoder.x

dscnn.x:
	g++ -std=c++17 -O3 driver.cpp -DMODEL=dscnn -o dscnn.x

resnet.x:
	g++ -std=c++17 -O3 driver.cpp -DMODEL=resnet -o resnet.x