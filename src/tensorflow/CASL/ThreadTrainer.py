# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
from threading import Thread
from Config import Config

class ThreadTrainer(Thread):
    def __init__(self, server, id):
        super(ThreadTrainer, self).__init__()
        self.setDaemon(True)
        self.id        = id
        self.server    = server
        self.exit_flag = False

    @staticmethod
    def _dynamic_pad(image_,audio_,r_,a_, lstm_prev_h_):
        t = image_.shape[0]
        if t != Config.TIME_MAX:
            imaget       = np.zeros((Config.TIME_MAX, Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH, Config.STACKED_FRAMES), dtype=np.float32)
            audiot       = np.zeros((Config.TIME_MAX, Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH, Config.STACKED_FRAMES), dtype=np.float32) 
            rt           = np.zeros((Config.TIME_MAX), dtype=np.float32)
            at           = np.zeros((Config.TIME_MAX, a_.shape[1]),dtype=np.float32)
            lstm_prev_ht = np.zeros((Config.TIME_MAX, Config.NCELLS), dtype=np.float32)

            imaget[:t] = image_; audiot[:t] = audio_; rt[:t] = r_; at[:t] = a_; lstm_prev_ht[:t] = lstm_prev_h_ # Fill from beginning to t with true image
            image_ = imaget; audio_ = audiot; r_ = rt; a_ = at; lstm_prev_h_ = lstm_prev_ht # Zero pad the suffix

        return image_, audio_, r_, a_, t, lstm_prev_h_

    def run(self):
        while not self.exit_flag:
            batch_size = 0
            seq_lengths__ = []

            if Config.USE_RNN:
                c__ = []; h__ = []
            else:
                c__ = None; h__ = None

            while batch_size <= Config.TRAINING_MIN_BATCH_SIZE:
                state_image_, state_audio_, r_, a_, rnn_state_, lstm_prev_h_ = self.server.training_q.get() # state_audio_ and rnn_state_ are None if not used
                if Config.USE_RNN:
                    state_image_, state_audio_, r_, a_, t, lstm_prev_h_ = ThreadTrainer._dynamic_pad(state_image_, state_audio_, r_, a_, lstm_prev_h_)
                    seq_lengths__.append(t)

                if batch_size == 0:
                    state_image__ = state_image_; state_audio__ = state_audio_; r__ = r_; a__ = a_; lstm_prev_h__ = lstm_prev_h_
                    if Config.USE_RNN and len(rnn_state_): 
                        c__ = []; h__ = []
                        for i in range(self.server.model.n_lstm_layers_total):
                            c, h = np.expand_dims(rnn_state_[i]['c'],0), np.expand_dims(rnn_state_[i]['h'],0)
                            c__.append(c)
                            h__.append(h)
                else:
                    state_image__ = np.concatenate((state_image__, state_image_))
                    r__ = np.concatenate((r__, r_))
                    a__ = np.concatenate((a__, a_))

                    if Config.USE_ATTENTION:
                        lstm_prev_h__ = np.concatenate((lstm_prev_h__, lstm_prev_h_))
                    else:
                        lstm_prev_h__ = None

                    if Config.USE_AUDIO: 
                        assert state_audio_ is not None
                        state_audio__ = np.concatenate((state_audio__, state_audio_))

                    if Config.USE_RNN and len(rnn_state_):
                        for i in range(self.server.model.n_lstm_layers_total):
                            c = np.expand_dims(rnn_state_[i]['c'],0)
                            h = np.expand_dims(rnn_state_[i]['h'],0)
                            c__[i] = np.concatenate((c__[i], c)) 
                            h__[i] = np.concatenate((h__[i], h)) # size of h is (n_lstm_layers_total, n_episodes_in_batch (usually floor(Config.TRAINIG_MIN_BATCH_SIZE / Config.TMAX), Config.NCELLS)

                batch_size += state_image_.shape[0]
            if Config.TRAIN_MODELS:
                self.server.train_model(state_image__, state_audio__, r__, a__, c__, h__, seq_lengths__, lstm_prev_h__)
