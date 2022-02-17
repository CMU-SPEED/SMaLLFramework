import tensorflow as tf
import tensorflow.keras.layers as nn
import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(6)

class FusedBlock(tf.keras.Model):
    def __init__(self, K, F_conv, S_conv, C_o_1, F_dw, S_dw):
        super().__init__()

        self.conv =  nn.Conv2D(
        K, F_conv, strides=(S_conv, S_conv), padding='valid',
        use_bias=False
        )
        self.pool = nn.Conv2D(
        K, F_dw, strides=(S_dw, S_dw), padding='valid',
        use_bias=False
        )

    @tf.function(experimental_compile=True)
    def call(self, input_tensor, training=False):
        out = self.conv(input_tensor)
        pool_out = self.pool(out)
        return pool_out