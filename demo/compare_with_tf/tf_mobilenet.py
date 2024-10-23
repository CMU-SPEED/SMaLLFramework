'''
MobilnetV1 from Silican Labs github page:
https://github.com/SiliconLabs/platform_ml_models/blob/master/eembc/Person_detection/mobilenet_v1_eembc.py
'''

import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, AveragePooling2D, MaxPooling2D
from tensorflow.keras.regularizers import l2

#define model
def DefModel(input_shape=[1,224,224,3],num_classes = 2):
    # Mobilenet parameters
    input_shape = input_shape[1:] # resized to 96x96 per EEMBC requirement
    num_classes = num_classes # person and non-person
    num_filters = 32 # was 8

    inputs = Input(shape=input_shape)
    x = inputs # Keras model uses ZeroPadding2D()
    layer_num = 1
    # 1st layer, pure conv
    # Keras 2.2 model has padding='valid' and disables bias
    x = Conv2D(num_filters,
                  kernel_size=3,
                  strides=2,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x) # Keras uses ReLU6 instead of pure ReLU
    # print("conv",layer_num, x.shape[1:])
    layer_num += 1
    # 2nd layer, depthwise separable conv
    # Filter size is always doubled before the pointwise conv
    # Keras uses ZeroPadding2D() and padding='valid'
    x = DepthwiseConv2D(kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # print("dwise conv",layer_num, x.shape[1:])
    layer_num += 1

    num_filters = 2*num_filters
    x = Conv2D(num_filters,
                  kernel_size=1,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # print("1x1 conv",layer_num, x.shape[1:])
    layer_num += 1

    # 3rd layer, depthwise separable conv
    x = DepthwiseConv2D(kernel_size=3,
                  strides=2,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # print("dwise conv",layer_num, x.shape[1:])
    layer_num += 1

    num_filters = 2*num_filters
    x = Conv2D(num_filters,
                  kernel_size=1,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # print("1x1 conv",layer_num, x.shape[1:])
    layer_num += 1

    # 4th layer, depthwise separable conv
    x = DepthwiseConv2D(kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # print("dwise conv",layer_num, x.shape[1:])
    layer_num += 1

    x = Conv2D(num_filters,
                  kernel_size=1,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # print("1x1 conv",layer_num, x.shape[1:])
    layer_num += 1

    # 5th layer, depthwise separable conv
    x = DepthwiseConv2D(kernel_size=3,
                  strides=2,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # print("dwise conv",layer_num, x.shape[1:])
    layer_num += 1

    num_filters = 2*num_filters
    x = Conv2D(num_filters,
                  kernel_size=1,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # print("1x1 conv",layer_num, x.shape[1:])
    layer_num += 1

    # 6th layer, depthwise separable conv
    x = DepthwiseConv2D(kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # print("dwise conv",layer_num, x.shape[1:])
    layer_num += 1

    x = Conv2D(num_filters,
                  kernel_size=1,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # print("1x1 conv",layer_num, x.shape[1:])
    layer_num += 1

    # 7th layer, depthwise separable conv
    x = DepthwiseConv2D(kernel_size=3,
                  strides=2,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # print("dwise conv",layer_num, x.shape[1:])
    layer_num += 1

    num_filters = 2*num_filters
    x = Conv2D(num_filters,
                  kernel_size=1,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # print("1x1 conv",layer_num, x.shape[1:])
    layer_num += 1

    # 8th-12th layers, identical depthwise separable convs
    # 8th
    x = DepthwiseConv2D(kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # print("dwise conv",layer_num, x.shape[1:])
    layer_num += 1

    x = Conv2D(num_filters,
                  kernel_size=1,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # print("1x1 conv",layer_num, x.shape[1:])
    layer_num += 1

    # 9th
    x = DepthwiseConv2D(kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # print("dwise conv",layer_num, x.shape[1:])
    layer_num += 1

    x = Conv2D(num_filters,
                  kernel_size=1,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # print("1x1 conv",layer_num, x.shape[1:])
    layer_num += 1

    # 10th
    x = DepthwiseConv2D(kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # print("dwise conv",layer_num, x.shape[1:])
    layer_num += 1

    x = Conv2D(num_filters,
                  kernel_size=1,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # print("1x1 conv",layer_num, x.shape[1:])
    layer_num += 1

    # 11th
    x = DepthwiseConv2D(kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # print("dwise conv",layer_num, x.shape[1:])
    layer_num += 1

    x = Conv2D(num_filters,
                  kernel_size=1,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # print("1x1 conv",layer_num, x.shape[1:])
    layer_num += 1

    # 12th
    x = DepthwiseConv2D(kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # print("dwise conv",layer_num, x.shape[1:])
    layer_num += 1

    x = Conv2D(num_filters,
                  kernel_size=1,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # print("1x1 conv",layer_num, x.shape[1:])
    layer_num += 1

    # 13th layer, depthwise separable conv
    x = DepthwiseConv2D(kernel_size=3,
                  strides=2,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # print("dwise conv",layer_num, x.shape[1:])
    layer_num += 1

    num_filters = 2*num_filters
    x = Conv2D(num_filters,
                  kernel_size=1,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # print("1x1 conv",layer_num, x.shape[1:])
    layer_num += 1

    # 14th layer, depthwise separable conv
    x = DepthwiseConv2D(kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # print("dwise conv",layer_num, x.shape[1:])
    layer_num += 1

    x = Conv2D(num_filters,
                  kernel_size=1,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # print("1x1 conv",layer_num, x.shape[1:])
    layer_num += 1

    # Average pooling, max polling may be used also
    # Keras employs GlobalAveragePooling2D 
    # x = AveragePooling2D(pool_size=x.shape[1:3])(x)
    x = MaxPooling2D(pool_size=x.shape[1:3])(x)
    # print("pool",layer_num, x.shape[1:])
    layer_num += 1

    # Keras inserts Dropout() and a pointwise Conv2D() here
    # We are staying with the paper base structure

    # Flatten, FC layer and classify
    x = Flatten()(x)
    # print("fc",layer_num, x.shape[1:])
    layer_num += 1

    outputs = Dense(num_classes)(x)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model



import time 


outputs = 16
input_shape = [1, 96, 96, 3]
model = DefModel(input_shape, outputs)

#set tensorflow to use GPU
tf.config.set_soft_device_placement(True)
tf.config.set_visible_devices([], 'GPU')





@tf.function(experimental_compile=True)
def model_inference(a):
    out = model(a)
    return out


a = np.random.randn(np.prod(input_shape)).reshape(input_shape)
# a[1:100] = 88
 
out = model_inference(a)
print(out)

print(model.summary())
# ULLONG_MAX=2**63
# sum_pool = ULLONG_MAX
# for r in range(1):
#     st = time.time()
#     for i in range(100):
#         out = model_inference(a)
#     et = time.time()
#     sum_pool = min(sum_pool,(et - st)/100)
#     # print((et-st))
# print(" \t {}".format(sum_pool))
