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
from Config import Config
from threading import Thread

class ThreadPredictor(Thread):
    def __init__(self, server, id):
        super(ThreadPredictor, self).__init__()
        self.setDaemon(True)
        self.id        = id
        self.server    = server
        self.exit_flag = False

    def run(self):
        ids = np.zeros(Config.PREDICTION_BATCH_SIZE, dtype=np.uint16)
        states_image = np.zeros((Config.PREDICTION_BATCH_SIZE, Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH, Config.STACKED_FRAMES), dtype=np.float32)
        states_audio = np.zeros((Config.PREDICTION_BATCH_SIZE, Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH, Config.STACKED_FRAMES), dtype=np.float32)

        if Config.USE_RNN:
            cs = np.zeros((Config.PREDICTION_BATCH_SIZE, self.server.model.n_lstm_layers_total, Config.NCELLS), dtype=np.float32) 
            hs = np.zeros((Config.PREDICTION_BATCH_SIZE, self.server.model.n_lstm_layers_total, Config.NCELLS), dtype=np.float32)
        else:
            cs = [None] * Config.PREDICTION_BATCH_SIZE
            hs = [None] * Config.PREDICTION_BATCH_SIZE

        while not self.exit_flag:
            ids[0], states_image[0], states_audio[0], cs[0], hs[0] = self.server.prediction_q.get()# Pops out the first element
            size = 1
            while size < Config.PREDICTION_BATCH_SIZE and not self.server.prediction_q.empty():
                ids[size], states_image[size], states_audio[size], cs[size], hs[size] = self.server.prediction_q.get()
                size += 1

            if Config.USE_RNN:
                if Config.USE_ATTENTION:
                    p, v, c, h, attention = self.server.model.predict_p_and_v(states_image[:size], states_audio[:size], cs[:size], hs[:size])# size is "batch size"
                else:
                    p, v, c, h = self.server.model.predict_p_and_v(states_image[:size], states_audio[:size], cs[:size], hs[:size])
            else:
                p, v = self.server.model.predict_p_and_v(states_image[:size], states_audio[:size], cs=None, hs=None)

            # Put p and v into wait_q accordingly
            for i in range(size):
                if ids[i] < len(self.server.agents):
                    if Config.USE_RNN:
                        assert c[i].shape == (self.server.model.n_lstm_layers_total, Config.NCELLS)
                        assert h[i].shape == (self.server.model.n_lstm_layers_total, Config.NCELLS)
                        if Config.USE_ATTENTION:
                            self.server.agents[ids[i]].wait_q.put((p[i], v[i], c[i], h[i], attention[i]))
                        else:
                            self.server.agents[ids[i]].wait_q.put((p[i], v[i], c[i], h[i]))
                    else:
                        self.server.agents[ids[i]].wait_q.put((p[i], v[i]))
