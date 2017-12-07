import os
import time
import re
import numpy as np
import tensorflow as tf
from models import CustomLayers 
from Config import Config

class NetworkVPCore(object):
    def __init__(self, device, num_actions):
        self.device        = device
        self.num_actions   = num_actions
        self.img_width     = Config.IMAGE_WIDTH
        self.img_height    = Config.IMAGE_HEIGHT
        self.img_channels  = Config.STACKED_FRAMES
        self.learning_rate = Config.LEARNING_RATE_START
        self.beta          = Config.BETA_START
        self.log_epsilon   = Config.LOG_EPSILON

        self.graph = tf.Graph()
        with self.graph.as_default() as g:
            with tf.device(self.device):
                self._create_graph()

                self.sess = tf.Session(
                    graph=self.graph,
                    config=tf.ConfigProto(
                        allow_soft_placement=True,
                        log_device_placement=False,
                        gpu_options=tf.GPUOptions(allow_growth=True)))
                self.sess.run(tf.global_variables_initializer())

                if Config.TENSORBOARD:
                    self._create_tensorboard()
                if Config.LOAD_CHECKPOINT or Config.SAVE_MODELS:
                    vars = tf.global_variables()
                    self.saver = tf.train.Saver({var.name: var for var in vars}, max_to_keep=5, keep_checkpoint_every_n_hours=1.0)

    def _assert_net_type(self, is_rnn_model, is_attention_model):
        if (not is_rnn_model and Config.USE_RNN) or (is_rnn_model and not Config.USE_RNN): 
            raise ValueError('User specific Config.USE_RNN: ' + str(Config.USE_RNN) + ', but selected Config.NET_ARCH: ' + str(Config.NET_ARCH))

        if (not is_attention_model and Config.USE_ATTENTION): # Second case not needed, since can turn attention on or off as long as model supports it 
            raise ValueError('User specific Config.USE_ATTENTION: ' + str(Config.USE_ATTENTION) + ', but selected Config.NET_ARCH: ' + str(Config.NET_ARCH))

    def _create_graph_inputs(self):
        self.episode       = tf.Variable(0, dtype=tf.int32, name='episode')
        self.x             = tf.placeholder(tf.float32, [None, self.img_height, self.img_width, self.img_channels], name='X')
        self.action_index  = tf.placeholder(tf.float32, [None, self.num_actions])
        self.layer_tracker = []

        if Config.USE_AUDIO:
            self.input_audio = tf.placeholder(tf.float32, [None, self.img_height, self.img_width, self.img_channels], name='input_audio')

        if Config.USE_RNN:
            self.seq_lengths = tf.placeholder(tf.int32, [None], name='seq_lengths')
            self.mask        = tf.placeholder(tf.float32, [None], name='mask')

            # All LSTM inputs/outputs. All are stored/restored properly as a unified list, so can have any complex LSTM architectures desired.
            self.rnn_state_in  = []  
            self.rnn_state_out = []
            self.n_lstm_layers_total = 0
        else:
            self.mask = 1.

        if Config.USE_ATTENTION:
            self.lstm_prev_h = tf.placeholder(tf.float32, [None, Config.NCELLS], name='lstm_prev_h')

    def _create_graph_outputs(self):
        # Cost: v 
        self.logits_v = tf.squeeze(tf.layers.dense(inputs=self.final_flat, units=1, use_bias=True, activation=None, name='logits_v'), axis=[1])
        self.y_r = tf.placeholder(tf.float32, [None], name='Yr')
        self.cost_v = 0.5*tf.reduce_sum(tf.square(self.y_r - self.logits_v)*self.mask, axis=0)/tf.reduce_sum(self.mask)

        # Cost: p 
        self.logits_p = tf.layers.dense(inputs=self.final_flat, units=self.num_actions, name='logits_p', activation=None)
        self.softmax_p = tf.nn.softmax(self.logits_p) 
        self.selected_action_prob = tf.reduce_sum(self.softmax_p*self.action_index, axis=1, name='selection_action_prob')

        self.cost_p_advant= tf.log(tf.maximum(self.selected_action_prob, self.log_epsilon)) * (self.y_r - tf.stop_gradient(self.logits_v)) # Stop_gradient ensures the value gradient feedback doesn't contribute to policy learning
        self.var_beta = tf.placeholder(tf.float32, name='beta', shape=[])
        self.cost_p_entrop = -1.*self.var_beta*tf.reduce_sum(tf.log(tf.maximum(self.softmax_p, self.log_epsilon))*self.softmax_p, axis=1)

        self.cost_p_advant_agg = tf.reduce_sum(self.cost_p_advant*self.mask, axis=0, name='cost_p_advant_agg')/tf.reduce_sum(self.mask)
        self.cost_p_entrop_agg = tf.reduce_sum(self.cost_p_entrop*self.mask, axis=0, name='cost_p_entrop_agg')/tf.reduce_sum(self.mask)
        self.cost_p = -(self.cost_p_advant_agg + self.cost_p_entrop_agg)

        # Cost: attention
        if Config.USE_ATTENTION:
            self.cost_attention = -1.*Config.BETA_ATTENTION*tf.reduce_sum(tf.log(tf.maximum(self.softmax_attention, self.log_epsilon)) * self.softmax_attention, axis=1)
            self.cost_attention_agg = -tf.reduce_sum(self.cost_attention*self.mask, axis=0, name='cost_attention_agg')/tf.reduce_sum(self.mask) # Negative since want to maximize entropy 
        
            self.cost_all = self.cost_p + self.cost_v + self.cost_attention_agg
        else:
            self.cost_all = self.cost_p + self.cost_v

        # Optimizer
        self.var_learning_rate = tf.placeholder(tf.float32, name='lr', shape=[])
        if Config.OPTIMIZER == Config.OPT_RMSPROP:
            self.opt = tf.train.RMSPropOptimizer(learning_rate=self.var_learning_rate,
                                                 decay=Config.RMSPROP_DECAY,
                                                 momentum=Config.RMSPROP_MOMENTUM,
                                                 epsilon=Config.RMSPROP_EPSILON)
        elif Config.OPTIMIZER == Config.OPT_ADAM:
            self.opt = tf.train.AdamOptimizer(learning_rate=self.var_learning_rate)
        else:
            raise ValueError('Invalid optimizer chosen! Check Config.py!')

        # Grad clipping
        self.global_step = tf.Variable(0, trainable=False, name='step')
        if Config.USE_GRAD_CLIP:
            self.opt_grad = self.opt.compute_gradients(self.cost_all)
            self.opt_grad_clipped = [(tf.clip_by_average_norm(g, Config.GRAD_CLIP_NORM),v) for g,v in self.opt_grad]
            self.train_op = self.opt.apply_gradients(self.opt_grad_clipped, global_step = self.global_step)
        else:
            self.train_op = self.opt.minimize(self.cost_all, global_step=self.global_step)

    def _create_tensorboard(self):
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)

        summaries.append(tf.summary.scalar("Pcost_advantage", self.cost_p_advant_agg))
        summaries.append(tf.summary.scalar("Pcost_entropy", self.cost_p_entrop_agg))
        summaries.append(tf.summary.scalar("Pcost", self.cost_p))
        summaries.append(tf.summary.scalar("Vcost", self.cost_v))
        summaries.append(tf.summary.scalar("cost_all", self.cost_all))
        summaries.append(tf.summary.scalar("LearningRate", self.var_learning_rate))
        summaries.append(tf.summary.scalar("Beta", self.var_beta))
        if Config.USE_ATTENTION:
            summaries.append(tf.summary.scalar("Attentioncost_entropy", self.cost_attention_agg))

        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram("weights_%s" % var.name, var))

        for layer in self.layer_tracker:
            summaries.append(tf.summary.histogram(layer.name, layer))

        summaries.append(tf.summary.histogram("activation_final_flat", self.final_flat))
        summaries.append(tf.summary.histogram("activation_v", self.logits_v))
        summaries.append(tf.summary.histogram("activation_p", self.softmax_p))

        self.summary_op = tf.summary.merge(summaries)
        self.log_writer = tf.summary.FileWriter(os.path.join(Config.LOGDIR), self.sess.graph)

    def __get_base_feed_dict(self):
        return {self.var_beta: self.beta, self.var_learning_rate: self.learning_rate}

    def assert_and_update_feed_dict(self, feed_dict, x, audio, r, a, c, h, seq_lengths, mask, lstm_prev_h):
        assert x is not None
        feed_dict.update({self.x: x, self.y_r: r, self.action_index: a})

        if Config.USE_AUDIO:
            assert audio is not None
            feed_dict.update({self.input_audio: audio})

        if Config.USE_RNN:
            assert c is not None
            assert h is not None
            assert seq_lengths is not None
            assert mask is not None

            seq_lengths = np.array(seq_lengths)
            feed_dict.update({self.seq_lengths: seq_lengths, self.mask: mask})
            for i in range(self.n_lstm_layers_total):
                cb = np.array(c[i]).reshape((-1, Config.NCELLS))
                hb = np.array(h[i]).reshape((-1, Config.NCELLS))
                cb = c[i]
                hb = h[i]
                feed_dict.update({self.rnn_state_in[i]: (cb, hb)})

        if Config.USE_ATTENTION:
            feed_dict.update({self.lstm_prev_h: lstm_prev_h})

    def get_global_step(self):
        return self.sess.run(self.global_step)

    def predict_single(self, x, audio):
        if Config.USE_AUDIO:
            return self.predict_p(x[None, :], audio[None, :])[0]
        else:
            return self.predict_p(x[None, :])[0]

    def predict_v(self, x, audio):
        feed_dict = {self.x: x} 
        if Config.USE_AUDIO:
            feed_dict.update({self.input_audio: audio})
        return self.sess.run(self.logits_v, feed_dict=feed_dict)

    def predict_p(self, x, audio):
        feed_dict = {self.x: x} 
        if Config.USE_AUDIO:
            feed_dict.update({self.input_audio: audio})
        return self.sess.run(self.softmax_p, feed_dict=feed_dict)

    def predict_p_and_v(self, x, audio, cs, hs):
        batch_size = x.shape[0]
        feed_dict = self.__get_base_feed_dict()

        assert x is not None
        feed_dict.update({self.x: x})

        if Config.USE_AUDIO:
            assert audio is not None
            feed_dict.update({self.input_audio: audio})     

        if Config.USE_RNN:
            assert cs is not None
            assert hs is not None

            seq_lengths = np.ones((x.shape[0],), dtype=np.int32) # Prediction done step-by-step, so seq_length is always 1
            feed_dict.update({self.seq_lengths: seq_lengths})

            # Populate RNN states
            for i in xrange(self.n_lstm_layers_total):
                c = cs[:, i, :] if i == 1 else cs[:, i]
                h = hs[:, i, :] if i == 1 else hs[:, i]
                feed_dict.update({self.rnn_state_in[i]: (c, h)})

            if Config.USE_ATTENTION:
                lstm_prev_h = hs[:, 0, :]
                feed_dict.update({self.lstm_prev_h: lstm_prev_h})
                p, v, rnn_state_out, attention = self.sess.run([self.softmax_p, self.logits_v, self.rnn_state_out, self.softmax_attention], feed_dict=feed_dict)
            else:
                p, v, rnn_state_out = self.sess.run([self.softmax_p,  self.logits_v, self.rnn_state_out], feed_dict=feed_dict)

            # Updated RNN states to be sent back (since next timestep will need likely them)
            c = np.zeros((batch_size, self.n_lstm_layers_total, Config.NCELLS), dtype=np.float32)
            h = np.zeros((batch_size, self.n_lstm_layers_total, Config.NCELLS), dtype=np.float32)
            for i in xrange(self.n_lstm_layers_total):
                c[:, i, :] = rnn_state_out[i].c
                h[:, i, :] = rnn_state_out[i].h

            if Config.USE_ATTENTION:
                return p, v, c, h, attention
            else:
                return p, v, c, h
        else:
            p, v = self.sess.run([self.softmax_p, self.logits_v], feed_dict=feed_dict)
            return p, v

    def create_loss_mask(self, seq_lengths):
        if Config.USE_RNN:
            seq_lengths_size = len(seq_lengths)
            mask= np.zeros((seq_lengths_size, Config.TIME_MAX), np.float32)
            for i_row in xrange(0, seq_lengths_size):
                mask[i_row,:seq_lengths[i_row]] = 1.
            return mask.flatten()
        else:
            return None

    def train(self, x, audio, y_r, a, c, h, seq_lengths, lstm_prev_h):
        mask = self.create_loss_mask(seq_lengths) # Create mask
        feed_dict = self.__get_base_feed_dict()
        self.assert_and_update_feed_dict(feed_dict, x, audio, y_r, a, c, h, seq_lengths, mask, lstm_prev_h)
        self.sess.run(self.train_op, feed_dict=feed_dict)

    def log(self, x, audio, y_r, a, c, h, seq_lengths, reward, roll_reward, lstm_prev_h):
        feed_dict = self.__get_base_feed_dict()
        self.assert_and_update_feed_dict(feed_dict, x, audio, y_r, a, c, h, seq_lengths, self.create_loss_mask(seq_lengths), lstm_prev_h)

        step, summary = self.sess.run([self.global_step, self.summary_op], feed_dict=feed_dict)
        self.log_writer.add_summary(summary, step)

        summary = tf.Summary(value=[tf.Summary.Value(tag="Reward", simple_value=reward)])
        self.log_writer.add_summary(summary, step)

        summary = tf.Summary(value=[tf.Summary.Value(tag="Roll_Reward", simple_value=roll_reward)])
        self.log_writer.add_summary(summary, step)

    def _checkpoint_filename(self):
        return os.path.join(Config.LOGDIR, 'checkpoints', 'network')

    def save(self, episode):
        episode_assign_op = self.episode.assign(episode)
        self.sess.run(episode_assign_op) # Save episode number in the checkpoint
        self.saver.save(self.sess, self._checkpoint_filename())

    def load(self):
        filename = tf.train.latest_checkpoint(os.path.dirname(self._checkpoint_filename()))

        if Config.LOAD_EPISODE > 0:
            filename = self._checkpoint_filename(Config.LOAD_EPISODE)

        try:
            self.saver.restore(self.sess, filename)
        except:
            raise ValueError('Error importing checkpoint! Are you sure checkpoint %s exists?' %self._checkpoint_filename())

        return self.sess.run(self.episode)

    def get_variables_names(self):
        return [var.name for var in self.graph.get_collection('trainable_variables')]

    def get_variable_value(self, name):
        return self.sess.run(self.graph.get_tensor_by_name(name))
