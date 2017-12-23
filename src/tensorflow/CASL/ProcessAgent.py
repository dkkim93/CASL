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
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from multiprocessing import Process, Queue, Value
from Config import Config
from Environment import Environment
from Experience import Experience

class ProcessAgent(Process):
    def __init__(self, model, id, prediction_q, training_q, episode_log_q, num_actions, stats):
        super(ProcessAgent, self).__init__()
        self.model                  = model
        self.id                     = id
        self.prediction_q           = prediction_q
        self.training_q             = training_q
        self.episode_log_q          = episode_log_q
        self.num_actions            = num_actions
        self.actions                = np.arange(self.num_actions)
        self.discount_factor        = Config.DISCOUNT
        self.wait_q                 = Queue(maxsize=1)
        self.exit_flag              = Value('i', 0)
        self.stats                  = stats
        self.last_vis_episode_num   = 0
        self.is_vis_training        = False
        self.debug_experience_issue = False

    @staticmethod
    def _accumulate_rewards(experiences, discount_factor, terminal_reward, done):
        reward_sum = terminal_reward # terminal_reward is called R in a3c paper

        returned_exp = experiences[:-1] # Returns all but final experience in most cases. Final exp saved for next batch. 
        leftover_term_exp = None # For special case where game finishes but with 1 experience longer than TMAX
        n_exps = len(experiences)-1 # Does n_exps-step backward updates on all experiences

        # Exception case for experiences length of 0
        if len(experiences) == 1:
            experiences[0].reward = np.clip(experiences[0].reward, Config.REWARD_MIN, Config.REWARD_MAX) 
            return experiences, leftover_term_exp 
        else:
            if done and len(experiences) == Config.TIME_MAX+1:
                leftover_term_exp = [experiences[-1]]
            if done and len(experiences) != Config.TIME_MAX+1:
                n_exps = len(experiences)
                returned_exp = experiences

            for t in reversed(xrange(0, n_exps)):
                # experiences[t].reward is single-step reward here
                r = np.clip(experiences[t].reward, Config.REWARD_MIN, Config.REWARD_MAX) 
                reward_sum = discount_factor * reward_sum + r 
                # experiences[t]. reward now becomes y_r (target reward, with discounting), and is used as y_r in training thereafter. 
                experiences[t].reward = reward_sum 

            return returned_exp, leftover_term_exp # NOTE Final experience is removed 

    def convert_to_nparray(self, experiences):
        x_ = np.array([exp.state_image for exp in experiences])
        a_ = np.eye(self.num_actions)[np.array([exp.action for exp in experiences], dtype=np.int32)].astype(np.float32)
        r_ = np.array([exp.reward for exp in experiences])

        if Config.USE_ATTENTION:
            lstm_prev_h_ = np.zeros((len(experiences), Config.NCELLS))
            for index, exp in enumerate(experiences):
                lstm_prev_h_[index, :] = exp.lstm_prev_h
        else:
            lstm_prev_h_ = None

        if Config.USE_AUDIO:
            audio_ = np.array([exp.state_audio for exp in experiences])
            return x_, audio_, r_, a_, lstm_prev_h_
        else:
            return x_, None, r_, a_, lstm_prev_h_

    def predict(self, current_state, rnn_state):
        if Config.USE_RNN:
            # lstm_inputs: [dict{stacklayer1}, dict{stacklayer2}, ...]
            assert rnn_state is not None
            c_state = np.array([lstm['c'] for lstm in rnn_state]) if len(rnn_state) else None
            h_state = np.array([lstm['h'] for lstm in rnn_state]) if len(rnn_state) else None
        else:
            c_state = None
            h_state = None

        if Config.USE_AUDIO:
            state_image = current_state[0]
            state_audio = current_state[1]
            assert state_image is not None
            assert state_audio is not None
        else:
            state_image = current_state
            state_audio = None
            assert state_image is not None

        # Put the state in the prediction q
        self.prediction_q.put((self.id, state_image, state_audio, c_state, h_state))

        if Config.USE_RNN:
            if Config.USE_ATTENTION:
                p, v, c_state, h_state, attention = self.wait_q.get() # wait for the prediction to come back
            else:
                p, v, c_state, h_state = self.wait_q.get() # wait for the prediction to come back
            if not len(rnn_state):
                return p, v, [] 

            # convert return back to form: [dict{stacklayer1}, dict{stacklayer2}, ...]
            l = [{'c': c_state[i], 'h': h_state[i]} for i in range(c_state.shape[0])]

            if Config.USE_ATTENTION:
                return p, v, l, attention
            else:
                return p, v, l
        else:
            p, v = self.wait_q.get() # wait for the prediction to come back
            return p, v

    def select_action(self, prediction):
        if Config.PLAY_MODE:
            action = np.argmax(prediction)
        else:
            action = np.random.choice(self.actions, p=prediction)

        return action

    def run_episode(self):
        self.env.reset()
        done              = False
        experiences       = []
        time_count        = 0
        iter_count        = 0
        reward_sum_logger = 0.0

        if Config.USE_RNN:
            # input states for prediction
            rnn_state = [{'c': np.zeros(Config.NCELLS, dtype=np.float32),
                          'h': np.zeros(Config.NCELLS, dtype=np.float32)}] * self.model.n_lstm_layers_total 

            # input states for training
            init_rnn_state = [{'c': np.zeros(Config.NCELLS, dtype=np.float32),
                               'h': np.zeros(Config.NCELLS, dtype=np.float32)}] * self.model.n_lstm_layers_total
        else:
            init_rnn_state = None

        if self.model.n_lstm_layers_total > 1:
            raise RuntimeError("Not implemented yet for n_lstm_layers_total > 1")

        while not done:
            # Initial step (used to ensure frame_q is full before trying to grab a current_state for prediction)
            if Config.USE_AUDIO:
                # Take null action until queue gets filled
                if self.env.current_state[0] is None and self.env.current_state[1] is None:
                    self.env.step(0)# Action 0 corresponds to null action 
                    continue
            else:
                if self.env.current_state is None:
                    self.env.step(0)# Action 0 corresponds to null action
                    continue

            # Prediction
            if Config.USE_RNN:
                if Config.USE_ATTENTION:
                    lstm_prev_h = rnn_state[0]['h']
                    prediction, value, rnn_state, attention = self.predict(self.env.current_state, rnn_state)
                else:
                    lstm_prev_h = None
                    prediction, value, rnn_state = self.predict(self.env.current_state, rnn_state)
            else:
                prediction, value = self.predict(self.env.current_state, rnn_state = None)

            # Visualize train process or test process
            if (self.id == 0 and self.is_vis_training) or Config.PLAY_MODE:
                if Config.USE_ATTENTION:
                    # Attention append
                    self.vis_attention_i.append(attention[0])
                    self.vis_attention_a.append(attention[1])
                else:
                    self.vis_attention_i = None
                    self.vis_attention_a = None

                self.env.visualize_env(self.vis_attention_i, self.vis_attention_a)

            # Select action
            action = self.select_action(prediction)

            # Take action --> Receive reward, done (and also store self.env.previous_state for access below)
            reward, done = self.env.step(action)
            reward_sum_logger += reward # Used for logging only

            # Add to experience
            if Config.USE_AUDIO:
                exp = Experience(self.env.previous_state[0], self.env.previous_state[1],
                                 action, prediction, reward, done, lstm_prev_h)
            else:
                exp = Experience(self.env.previous_state, None,
                                 action, prediction, reward, done, lstm_prev_h)
            experiences.append(exp)

            # If episode is done
            # Config.TIME_MAX controls how often data is yielded/sent back to the for loop in the run(). 
            # It is used to ensure, for games w long episodes, that data is sent back to the trainers sufficiently often
            # The shorter Config.TIME_MAX is, the more often the data queue is updated 
            if done or time_count == Config.TIME_MAX:
                terminal_reward = 0 if done else value # A3C paper, Algorithm S2 (n-step q-learning) 
                updated_exps, updated_leftover_exp = ProcessAgent._accumulate_rewards(experiences, self.discount_factor, terminal_reward, done)
                x_, audio_, r_, a_, lstm_prev_h_ = self.convert_to_nparray(updated_exps) # NOTE if Config::USE_AUDIO == False, audio_ is None
                yield x_, audio_, r_, a_, init_rnn_state, reward_sum_logger, lstm_prev_h_ # Sends back data and starts here next time fcn is called

                reward_sum_logger = 0.0 # NOTE total_reward_logger in self.run() accumulates reward_sum_logger, so it is correct to reset it here 

                if updated_leftover_exp is not None:
                    #  terminal_reward = 0
                    x_, audio_, r_, a_, lstm_prev_h_ = self.convert_to_nparray(updated_leftover_exp) # NOTE if Config::USE_AUDIO == False, audio_ is None
                    yield x_, audio_, r_, a_, init_rnn_state, reward_sum_logger, lstm_prev_h_ 

                # Reset the tmax count
                time_count = 0

                # Keep the last experience for the next batch
                experiences = [experiences[-1]]

                if Config.USE_RNN:
                    init_rnn_state = rnn_state 

            time_count += 1
            iter_count += 1

    def run(self):
        # Randomly sleep up to 1 second. Helps agents boot smoothly.
        time.sleep(np.random.rand())
        np.random.seed(np.int32(time.time() % 1 * 5000 + self.id * 10))

        self.env = Environment() # NOTE env is created in here

        while self.exit_flag.value == 0:
            total_reward_logger = 0
            total_length        = 0

            # For visualizing train process
            if self.id == 0 and Config.VIS_TRAIN:
                self.current_episode_num = self.stats.episode_count.value
                if ((self.current_episode_num - self.last_vis_episode_num > Config.VIS_FREQUENCY)) or Config.PLAY_MODE:
                    self.is_vis_training = True
                    if Config.USE_ATTENTION:
                        self.vis_attention_i = []
                        self.vis_attention_a = []

            for x_, audio_, r_, a_, rnn_state_, reward_sum_logger, lstm_prev_h_ in self.run_episode():
                if len(x_.shape) <= 1:
                    raise RuntimeError("x_ has invalid shape")
                total_reward_logger += reward_sum_logger
                total_length += len(r_) + 1  # +1 for last frame that we drop
                self.training_q.put((x_, audio_, r_, a_, rnn_state_, lstm_prev_h_)) # NOTE audio_ and rnn_state_ might be None depending on Config.USE_AUDIO/USE_RNN

            self.episode_log_q.put((datetime.now(), total_reward_logger, total_length))

            # Close visualizing train process
            if (self.id == 0 and self.is_vis_training) or Config.PLAY_MODE:
                self.is_vis_training = False
                self.last_vis_episode_num = self.current_episode_num

