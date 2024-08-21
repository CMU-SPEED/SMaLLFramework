This document walks you through the process of adding a new layer 

Adding a new layer that uses the abstract layer requires modifications at 4 levels
- Top Level: To be compatible with the model constructor
	- Create a a file for your layer in `include/small/<your_layer_name>Layer.hpp` following the template in `include/small/Conv2DLayer.hpp` . Call the appropriate function from interface_abstract.hpp in the `compute_output()` method of the Layer. 
- Higher Level Implementation Specification:
	- Add a declaration of the new layer in the small namespace in `include/small/interface.hpp`
	- Add a definition following the signature declared above in  `include/small/interface_abstract.hpp`. The definition will call the `abstract_layer` with the appropriate parameters
		- The parameters
- `include/small/abstract_layer.hpp`
	-  If needed, the definition of `ABSTRACT_OP_C` and `ABSTRACT_OP_END` to use the computation kernel provided in `include/small/platforms/?/intrinsics.h` 
	- If needed, change the `LOAD` and `STORE` kernels.
- `include/small/platforms/*/intrinsics.h`
	- Implement a `LOAD`, `COMPUTE` and `STORE` phase for the new layer
