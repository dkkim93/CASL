import os, re
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
from NetworkVPCore import NetworkVPCore
from models import CustomLayers
from Config import Config

class Net_3C_XL_A_F(NetworkVPCore):
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

        # Video LSTM
        rnn_out_i, self.n_lstm_layers_total, self.hmm = CustomLayers.multilayer_lstm(input = fc1_i, 
                                                                                     n_lstm_layers_total=self.n_lstm_layers_total,
                                                                                     global_rnn_state_in=self.rnn_state_in, 
                                                                                     global_rnn_state_out=self.rnn_state_out, 
                                                                                     base_name='i_', 
                                                                                     seq_lengths=self.seq_lengths)

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

            # Audio LSTM
            rnn_out_a, self.n_lstm_layers_total, _ = CustomLayers.multilayer_lstm(input = fc1_a, 
                                                                                  n_lstm_layers_total=self.n_lstm_layers_total,
                                                                                  global_rnn_state_in=self.rnn_state_in, 
                                                                                  global_rnn_state_out=self.rnn_state_out, 
                                                                                  base_name='a_', 
                                                                                  seq_lengths=self.seq_lengths)

        # Attention
        if Config.USE_ATTENTION:
            fc_dim = 256
            fused_layer, self.softmax_attention = CustomLayers.multimodal_attention_layer(input_i=rnn_out_i, attention_feat_i=rnn_out_i, 
                                                                                          input_a=rnn_out_a, attention_feat_a=rnn_out_a,
                                                                                          fc_dim=fc_dim, 
                                                                                          fusion_mode=CustomLayers.FUSION_SUM)
            self.layer_tracker.append(tf.layers.dense(inputs=fused_layer, units=fc_dim, use_bias=True, activation=tf.nn.relu, name='fc2'))
        else:
            self.layer_tracker.append(tf.concat([rnn_out_i, rnn_out_a], axis=1)) # axis = 0 would concat batches instead 

        # Output to NetworkVP
        self.final_flat = self.layer_tracker[-1] # Final layer must always be be called final_flat

        # -------- End custom architecture here --------
        
        # Use shared parent class to construct graph outputs/objectives
        self._create_graph_outputs()
