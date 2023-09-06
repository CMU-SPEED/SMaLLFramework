# Defining New Platforms

## Introduction

This document outlines what is required to define a new platform in the SMaLLFramework.

### Definition of a platform

A platform consists of a piece of hardware and possibly a specific version of a tool chain to build applications on that platform.

This document mainly describes the `reference` platform which assumes vanilla hardware.  This platform is also provided as an example of one that has support for two different data types: 32-bit floating point and 8-bit quantized integers.

### File organization and Build system

The platform specific files must all appear in a subdirectory of `include\small\platforms`.  The name of the directory will be referred to as the name of the platform.

In order to provide build support for this new platform, the top-level CMakeLists.txt file must be modified to include a block that specifies the use of the files in this new directory.  Shown here for the `reference` platform:

```
if(CMAKE_UARCH STREQUAL "REF")
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -lrt -DUARCH_REF")
  include_directories(${CMAKE_PLATFORM_DIR}/reference)
  message("Configuring 'reference' platform.")
endif()
```

This block is used to define any platform specific compiler flags (TBD: determine if -DUARCH_<platform> is necessary anymore) and set the include path to include the platform directory.

### Feature test macros for type support

Each platform must define one feature test macro for each data type supported by that platform.  The following is an example of the format for a platform that supports floating point data as well as quantized, 8-bit integers:

```
#define SMALL_HAS_FLOAT_SUPPORT 1
#define SMALL_HAS_QUINT8_SUPPORT 1
```

## Platform-specific files

Platform-specific code consists of sets of three files:
* `params.h` - Contains a set of constants that control block sizes that are tied to hardware capabilities for a specific data type.
* `Buffer.hpp` - Defines a class that manages a block of data of a single data type (e.g. `float`)
* `intrinsics.hpp` - Defines a set of macros that implement all of the microkernels for a given platform and data type.

One platform may support multiple data types such as single precision floating point (`float`) and/or 8-bit quantized integers (`quint8`) (see 'reference' platform).  In this case, a separate set of headers files may be defined for each data type and the files listed above contain include directives to include each type specific file.

For example, in the `reference` platform:
* `params.h` includes `params_float.h` and `params_quint.h`
* `Buffer.hpp` includes `FloatBuffer.hpp` and `QUInt8Buffer.hpp`
* `intrinsics.h` includes `intrinsics_float.h` and `intrinsics_quint8.h`

### Platform-Specific Constants

This file must define a set of macros (compile-time constants) that are selected to tune the loop nests to various hardware capabilities (TODO: FILL IN BETTER DEFINITION).  Note that each macro must have a prefix corresponding to the data type it is used with.  For example, `FLOAT_` prefix is used for `float` types and `QUINT8_` is used for quantized unsigned 8-bit integer types.

* <TYPE>_W_ob -   (FILL IN: some sort of output blocking factor)
* <TYPE>_C_ob -   Output channel blocking factor
* <TYPE>_C_ib -   Input channel blocking factor (currently this must be the same as the output channel blocking factor).
* <TYPE>_SIMD -   ???
* <TYPE>_UNROLL - ???
* <TYPE>_NUM_FMA - ??? (is this used?)
* <TYPE>_NUM_MAX - ??? (is this used?)
* <TYPE>_NUM_LOAD - ??? (is this used?)
* <TYPE>_NUM_STORE - ??? (is this used?)

### Support for Buffers of different data types

#### Dynamic allocation and deallocation

This section is TBD:

Two free functions need to be defined in a top-level Buffer.hpp file:

* A factory function templated on the Buffer type:
```
template <class BufferT>
inline BufferT *alloc_buffer(size_t num_elements);
```
* A way to deallocate the buffers created by the factory:
```
template <class BufferT>
inline void free_buffer(BufferT *buffer)
```

TBD: We believe these are needed abstract away how memory is allocated for systems like the Arduino where only static allocation is allowed.

#### Buffer classes for specific types

Untemplated classes are used to declare a different Buffer class for each data type supported by the platform.

The convention used is that FloatBuffer class is declared for floating point types, QUInt8Buffer is used for quantized, unsigned 8-bit integers.

##### Platform constants (again)

Each buffer class must contain static const members for all of the values defined in the corresponding `params_<type>.h` file:

```
class FloatBuffer
{
public:
    static uint32_t const   W_ob{FLOAT_W_ob};
    static uint32_t const   C_ob{FLOAT_C_ob};
    static uint32_t const   SIMD{FLOAT_SIMD};
    static uint32_t const UNROLL{FLOAT_UNROLL};
    static uint32_t const   C_ib{FLOAT_C_ib};

    static uint32_t const   NUM_FMA{FLOAT_NUM_FMA};
    static uint32_t const   NUM_MAX{FLOAT_NUM_MAX};
    static uint32_t const  NUM_LOAD{FLOAT_NUM_LOAD};
    static uint32_t const NUM_STORE{FLOAT_NUM_STORE};
};
```

Note that the prefix must be removed so that each `<Type>Buffer` class defines exactly the same set of constants.

TBD: make sure these are compile-time constants.

##### Buffer Class typedefs

Two class typedefs must be defined:

```
class FloatBuffer
{
public:
    typedef float value_type;
    typedef float accum_type;
};
```

*`value_type` - type of data stored in the buffer
*`accum_type` - mainly used by quantized types to define the type used to accumulate quantized value (usually a larger integer type)

##### Buffer Class Methods API

* `<Type>Buffer()` default constructor - defines a buffer that has no storage, zero elements.
* `<Type>Buffer(size_t num_elements)` constructor - creates an instance of a buffer with storage for the specified number of elements.
* Copy constructor
* Move constructor
* Copy assignment operator
* Move assignment operator
* Destructor (not virtual)
* `size_t size() const;` - return the number of elements in the buffer
* `value_type [const] *data() [const];` - return pointer to beginning of buffer memory
* `value_type [const] &operator[](size_t index) [const];` - index into array using square bracket.
* `void swap(<Type>Buffer &other);` - swap the contents of this buffer with `other`.
* `accum_type zero() const;' - mainly used by quantized types to report the bit representation of a zero value.

### Microkernels

All microkernels are implemented in macros.  Because microkernels can be different for different datatypes on one platform, these macros must also contain a type prefix.

#### Variable Declarations and Initializations

* <TYPE>_DEF_TILE_C
* <TYPE>_DEF_END_C
* <TYPE>_ZERO_TILE_C
* <TYPE>_ZERO_END_C

#### Loads

* <TYPE>_LOAD_TILE_C
* <TYPE>_LOAD_END_C

#### Pooling Loads

* <TYPE>_LOAD_TILE_C_strided
* <TYPE>_LOAD_END_C_strided

#### Upsampling Loads

* <TYPE>_LOAD_TILE_C_upsample
* <TYPE>_LOAD_END_C_upsample

#### Loads

* <TYPE>_STORE_TILE_C
* <TYPE>_STORE_END_C

For quantized types only: 

* <QTYPE>_STORE_Q_TILE_C
* <QTYPE>_STORE_Q_END_C

#### Convolution

* <TYPE>_CONV_TILE_C
* <TYPE>_CONV_END_C

#### Depthwise Convolution

* <TYPE>_DW_TILE_C
* <TYPE>_DW_END_C

#### Max Pooling and ReLU

* <TYPE>_MAX_TILE_C
* <TYPE>_MAX_END_C

#### Leaky ReLU

Conditional scaling

* <TYPE>_COND_SCALE_TILE_C
* <TYPE>_COND_SCALE_END_C

#### Accumulation kernels

* <TYPE>_ACCUM_TILE_C
* <TYPE>_ACCUM_END_C

#### Average Pooling

* <TYPE>_ADD_TILE_C_G
* <TYPE>_ADD_END_C_G

???

* <TYPE>_REDUCE_div_C
* <TYPE>_REDUCE_C
* <TYPE>_REDUCE_C_LAST

#### For Quantized types only

* VQRDMULH
* RNDRSHIFT
* <QTYPE>_QUANTIZE_TILE_C
* <QTYPE>_QUANTIZE_END_C

## Abstract layers (`abstract_layer.hpp`)

Loop nests are defined in `abstract_layer.hpp` and the code within makes calls to the various macros just discussed depending on the operations performed. There are currently two implementations of the abstract_layer loop nest: one for common numerical types and one for quantized types.  The former appears in the `small::detail` namespace and the latter appears in the (currently) `small::quint8_detail` namespace.

TBD: if the latter can also be used with the qint8 data type then the corresponding namespace will be renamed to `small::q_detail`

## Function Call Implementation (`interface.hpp`)

The function call interface is declared in `interface.hpp` and defined in `interfaceabstract.hpp` contains templated functions for each neural network layer type supported by SMaLLFramework.  The signature of these functions should not need to change.  An example of one function is:
```
template <class BufferT>
void Conv2D(int kernel_size, int stride,
            uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
            int output_channels, int input_channels,
            int input_height, int input_width,
            BufferT const &input_buf,
            BufferT const &filter_buf,
            BufferT       &output_buf);
```

This function is explicitly specialized for the Buffer type in order to call the correct abstract_layer() function (quantized or otherwise).  For `FloatBuffer` this looks like:
```
#if defined(SMALL_HAS_FLOAT_SUPPORT)
template <>
void Conv2D<FloatBuffer>(
    int kernel_size, int stride,
    uint8_t t_pad, uint8_t b_pad, uint8_t l_pad, uint8_t r_pad,
    int output_channels, int input_channels,
    int input_height, int input_width,
    FloatBuffer const &input_buf,
    FloatBuffer const &filter_buf,
    FloatBuffer       &output_buf)
{
    // ...
    
            detail::abstract_layer<
                FloatBuffer, 1, FLOAT_C_ob, FLOAT_C_ib,
                FLOAT_W_ob, 1, FLOAT_UNROLL, 'c', 2, 1>(
                    1,               // Output Channel Grouping
                    output_channels, // Output Channels per group
                    input_channels,
                    input_height, input_width,
                    kernel_size, kernel_size,
                    t_pad, l_pad, r_pad, b_pad,
                    &input_buf, &filter_buf, &output_buf);
    // ...
}
#endif
```
Note:
* The use of the feature test macro (`SMALL_HAS_FLOAT_SUPPORT`).
* The explicit specialization for the corresponding buffer type (`FloatBuffer`).
* The call to `abstract_layer()` in the correct namespace (`detail`)
