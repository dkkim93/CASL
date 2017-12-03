import tensorflow as tf
from Config import Config
from tensorflow.contrib import rnn
from tensorflow.contrib.cudnn_rnn import CudnnLSTM

# Attention mode enums
FUSION_SUM, FUSION_CONC = range(2)

def multilayer_cnn(input, n_conv_layers, layer_tracker, filters, kernel_size, strides, use_bias, padding, activation, base_name):
    for i_conv in xrange(0, n_conv_layers):
        layer_tracker.append(tf.layers.conv2d(inputs=input, filters=filters, kernel_size=kernel_size, strides=strides, use_bias=use_bias, padding=padding, activation=activation, name= '%s%s' % (base_name,str(i_conv))))
        input = layer_tracker[-1]
    return tf.contrib.layers.flatten(layer_tracker[-1])

def transpose_batch_time(x):
    x_static_shape = x.get_shape()
    if x_static_shape.ndims is not None and x_static_shape.ndims < 2:
        raise ValueError("Expected input tensor %s to have rank at least 2, but saw shape: %s" % (x, x_static_shape))

    x_rank = tf.rank(x)
    x_t = tf.transpose(x, tf.concat(([1, 0], tf.range(2, x_rank)), axis=0))
    x_t.set_shape(tf.TensorShape([x_static_shape[1].value, x_static_shape[0].value]).concatenate(x_static_shape[2:]))

    return x_t

def lstm_layer(input, input_dim, out_dim, initial_state_input, seq_lengths, name, reuse=False):
    if Config.USE_CUDNN + Config.USE_LSTMBlockFusedCell + Config.USE_LSTMCell > 1:
        raise ValueError('Only one of LSTM layer flags must be set to true in Config.py! But Config.USE_CUDNN=%i, Config.USE_LSTMBlockFusedCell=%i, Config.USE_LSTMCell=%i' % ( Config.USE_CUDNN, Config.USE_LSTMBlockFusedCell, Config.USE_LSTMCell))

    with tf.variable_scope(name, reuse=reuse):
        batch_size = tf.shape(seq_lengths)[0]
        input_reshaped = tf.reshape(input, [batch_size, -1, input_dim])

        if Config.USE_LSTMBlockFusedCell:
            cell = rnn.LSTMBlockFusedCell(num_units=out_dim)  # LSTMBlockFusedCell only supports tuple state 
            input_reshaped = transpose_batch_time(input_reshaped) # (B,T,I) -> (T,B,I) for LSTMBlockFusedCell 
            outputs, state = cell(inputs=input_reshaped,
                                  initial_state=initial_state_input,
                                  sequence_length=seq_lengths)
            outputs = transpose_batch_time(outputs) # (T,B,I) -> (B,T,I)

        elif Config.USE_LSTMCell:
            cell = rnn.LSTMCell(num_units=out_dim, state_is_tuple=True)  
            outputs, state = tf.nn.dynamic_rnn(cell,
                                               input_reshaped,
                                               initial_state=initial_state_input,
                                               sequence_length=seq_lengths,
                                               time_major=False)
        outputs = tf.reshape(outputs, [-1, out_dim])

        return outputs, state, batch_size

def multilayer_lstm(input, n_lstm_layers_total, global_rnn_state_in, global_rnn_state_out, base_name, seq_lengths):
    local_rnn_state_in  = []
    local_rnn_state_out = []

    for lstm_count in xrange(Config.NUM_LAYERS_PER_LSTM):
        input_dim = input.get_shape()[1].value # [0] is batch size, [1] is feature size
        c0 = tf.placeholder(tf.float32, [None, Config.NCELLS], '%s%s%s%s' % ('rnn_', base_name, 'c0_', str(lstm_count))) # add name prefix
        h0 = tf.placeholder(tf.float32, [None, Config.NCELLS], '%s%s%s%s' % ('rnn_', base_name, 'h0_', str(lstm_count))) # add name prefix
        local_rnn_state_in.append((c0,h0))

        rnn_out, rnn_state, batch_size = lstm_layer(input, input_dim, Config.NCELLS, rnn.LSTMStateTuple(c0, h0), seq_lengths, '%s%s%s' % ('rnn_', base_name, str(lstm_count)))
        local_rnn_state_out.append(rnn_state)
        input = rnn_out

    n_lstm_layers_total += Config.NUM_LAYERS_PER_LSTM
    global_rnn_state_in.extend(local_rnn_state_in)
    global_rnn_state_out.extend(local_rnn_state_out)

    return rnn_out, n_lstm_layers_total, batch_size

def multimodal_attention_layer(input_i, input_a, input_h, attn_dim, fusion_mode):
    assert fusion_mode in [FUSION_SUM, FUSION_CONC]

    linear_i = tf.layers.dense(inputs=input_i, units=attn_dim, activation=None, name='linear_i')
    linear_a = tf.layers.dense(inputs=input_a, units=attn_dim, activation=None, name='linear_a')
    linear_h = tf.layers.dense(inputs=input_h, units=attn_dim, activation=None, name='linear_h')

    tanh_layer = tf.add(tf.add(linear_i, linear_a), linear_h)
    tanh_layer = tf.layers.dense(inputs=tanh_layer, units=attn_dim, activation=tf.tanh, name='tanh_layer')

    softmax_attention = tf.layers.dense(inputs=tanh_layer, units=2, activation=tf.nn.softmax, name='softmax_attention')

    feat_i_attention = tf.multiply(input_i, tf.reshape(softmax_attention[:,0], [-1,1]), name='feat_i_attention')
    feat_a_attention = tf.multiply(input_a, tf.reshape(softmax_attention[:,1], [-1,1]), name='feat_a_attention')

    if fusion_mode == FUSION_SUM:
        fused_layer = tf.add(feat_i_attention, feat_a_attention, name='fused_layer')
    elif fusion_mode == FUSION_CONC:
        fused_layer = tf.concat([feat_i_attention, feat_a_attention], axis=1, name='fused_layer')

    return fused_layer, softmax_attention   
