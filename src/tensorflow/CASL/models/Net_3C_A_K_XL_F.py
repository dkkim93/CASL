import os, re
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
from NetworkVPCore import NetworkVPCore
from models import CustomLayers
from Config import Config

class Net_3C_A_K_XL_F(NetworkVPCore):
    def __init__(self, device, num_actions):
        super(self.__class__, self).__init__(device, num_actions)

    def _create_graph(self):
        self._assert_net_type(is_rnn_model = True, is_attention_model = True)

        # Use shared parent class to construct graph inputs
        self._create_graph_inputs()

        # -------- Put custom architecture here --------

        # Video CNN
        fc1_i = CustomLayers.multilayer_cnn(
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

        # Attention
        fusion_mode = CustomLayers.FUSION_CONC
        if Config.USE_ATTENTION:
            attn_dim = Config.NCELLS
            fused_layer, self.softmax_attention = CustomLayers.multimodal_attention_layer(input_i = fc1_i, input_a = fc1_a, input_h = self.lstm_prev_h,
                                                                                          attn_dim = attn_dim, fusion_mode = fusion_mode)
        else:
            if fusion_mode == CustomLayers.FUSION_SUM:
                fused_layer = tf.add(fc1_i, fc1_a, name = 'fused_layer')
            elif fusion_mode == CustomLayers.FUCION_CONC:
                fused_layer = tf.concat([fc1_i, fc1_a], axis = 1, name = 'fused_layer')

        fc_dim = 256
        self.layer_tracker.append(tf.layers.dense(inputs=fused_layer, units=fc_dim, use_bias = True, activation=tf.nn.relu, name='fc2'))

        # LSTM
        rnn_out, self.n_lstm_layers_total, _ = CustomLayers.multilayer_lstm(
                                                                            input = self.layer_tracker[-1], 
                                                                            n_lstm_layers_total = self.n_lstm_layers_total, 
                                                                            global_rnn_state_in = self.rnn_state_in, 
                                                                            global_rnn_state_out = self.rnn_state_out, 
                                                                            base_name = '', 
                                                                            seq_lengths = self.seq_lengths
                                                                           )

        # Output to NetworkVP
        self.final_flat = rnn_out # Final layer must always be be called final_flat

        # -------- End custom architecture here --------
        
        # Use shared parent class to construct graph outputs/objectives
        self._create_graph_outputs()
