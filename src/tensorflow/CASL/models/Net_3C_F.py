import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
from NetworkVPCore import NetworkVPCore
from models import CustomLayers
from Config import Config

class Net_3C_F(NetworkVPCore):
    def __init__(self, device, num_actions):
        super(self.__class__, self).__init__(device, num_actions)

    def _create_graph(self):
        self._assert_net_type(is_rnn_model = False, is_attention_model = False)

        # Use shared parent class to construct graph inputs
        self._create_graph_inputs()

        # -------- Put custom architecture here --------
        # Video CNN
        fc1  = CustomLayers.multilayer_cnn(
                input         = self.x,
                n_conv_layers = 3,
                layer_tracker = self.layer_tracker,
                filters       = 32,
                kernel_size   = [3,3],
                strides       = [2,2],
                use_bias      = True,
                padding       = "SAME",
                activation    = tf.nn.relu,
                base_name     = 'conv_v_'
            )

        # Audio CNN
        if Config.USE_AUDIO:
            fc1_a = CustomLayers.multilayer_cnn(
                    input         = self.input_audio,
                    n_conv_layers = 3,
                    layer_tracker = self.layer_tracker,
                    filters       = 32,
                    kernel_size   = [3,3],
                    strides       = [2,2],
                    use_bias      = True,
                    padding       = "SAME",
                    activation    = tf.nn.relu, 
                    base_name = 'conv_a_'
                )
            fc1  = tf.concat([fc1, fc1_a], axis=1) # axis = 0 would concat batches instead 

        # Final layer must always be be called final_flat
        self.final_flat = tf.layers.dense(fc1, units=256, use_bias=True, activation=tf.nn.relu, name='fc1')
        # -------- End custom architecture here --------
        
        # Use shared parent class to construct graph outputs/objectives
        self._create_graph_outputs()
