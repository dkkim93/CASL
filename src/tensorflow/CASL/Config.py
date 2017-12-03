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
import os
from datetime import datetime

class Config:
    #########################################################################
    # GENERAL PARAMETERS
    CODE_DIR            = os.path.dirname(__file__)
    game_doorpuzzle, 
    game_minecraft,
    game_ale            = range(3)  # Initialize game types as enum
    GAME_CHOICE         = game_ale
    USE_AUDIO           = True      # Enable audio input
    TRAIN_MODELS        = False     # Enable to train
    PLAY_MODE           = True      # Enable to see the trained agent in action (for testing)
    LOAD_EPISODE        = 0         # If 0, the latest checkpoint is loaded
    LOAD_CHECKPOINT     = True      # Load old models. Throws if the model doesn't exist
    VIS_TRAIN           = False     # Visualize train process
    VIS_FREQUENCY       = 500       # Enable Vis during trainig every VIS_FREQUENCY
    if LOAD_CHECKPOINT:
        LOGDIR          = os.path.join('tmp-logs', '2017-10-30_20h-12m-47s')
    else:
        LOGDIR          = os.path.join('tmp-logs', datetime.now().strftime("%Y-%m-%d_%Hh-%Mm-%Ss"))

    #########################################################################
    # MAZEWORLD GAME PARAMETER
    if GAME_CHOICE == game_doorpuzzle:
        REWARD_MIN     = -1    # Reward Clipping
        REWARD_MAX     = 1     # Reward Clipping
        ENV_ROW        = 5     # Number of patch in row direction
        ENV_COL        = 5     # Number of patch in col direction
        PIXEL_SIZE     = 20    # Pixel size for each patch
        MAX_ITER       = 30    # Max iteration (time limit)
        TIMER_DURATION = 0.001 # In second visualization time for each step
        NOISE_TRANS    = 0.2   # P(not going in dir of chosen action) 
        LISTEN_RANGE   = 1.5   # Distance needed to listen to a target
        HARD_MODE      = False # Gate and target positions become random
        if TRANSFER_MODE:
            HARD_MODE  = True  # If TRANSFER_MODE, set harder environment
        SIMPLE_RENDER  = True  # Option whether to show simple vis or more complex vis with characters, items, etc

    # MINECRAFT GAME PARAMETER
    elif GAME_CHOICE == game_minecraft:
        REWARD_MIN     = -100  # Reward Clipping
        REWARD_MAX     = 100   # Reward Clipping
        ENV_ROW        = 5     # Number of patch in row direction
        ENV_COL        = 5     # Number of patch in col direction
        PIXEL_SIZE     = 20    # Pixel size for each patch
        MAX_ITER       = 30    # Max iteration (time limit)
        LISTEN_RANGE   = 1.5   # Distance needed to listen to a target
        TIMER_DURATION = 0.001 # In second visualization time for each step
        SIMPLE_RENDER  = False # Option whether to show simple vis or more complex vis with characters, items, etc

    # ALE GAME PARAMETER
    elif GAME_CHOICE == game_ale:
        NUM_ACTIONS    = 10 # NOTE Depends on game (pong:6, seaquest: 18, amidar: 10)
        ATARI_GAME     = 'amidar'# Name of the game, with version (e.g. PongDeterministic-v0)
        REWARD_MIN     = -1 # Reward Clipping
        REWARD_MAX     = 1
        AUDIO_MAX_DATA = 512 

    #########################################################################
    # NET ARCHITECTURE
    # Possible NET_ARCH: 'Net_3C_A_K_XL_F', 'Net_3C_XL_A_F', 'Net_3C_F'
    NET_ARCH                   = 'Net_3C_A_K_XL_F' # Neural net architecture. Any from the 'models' folder can be selected.
    USE_RNN                    = True
    USE_ATTENTION              = True
    if USE_RNN:
        NUM_LAYERS_PER_LSTM    = 1
        NCELLS                 = 128
        STACKED_FRAMES         = 1
        USE_LSTMBlockFusedCell = True  # Uses LSTMBlockFusedCell tensorflow implementation
        USE_LSTMCell           = False # Uses LSTMCell tensorflow implementation
    else:
        STACKED_FRAMES         = 4

    #########################################################################
    # NUMBER OF AGENTS, PREDICTORS, TRAINERS, AND OTHER SYSTEM SETTINGS
    # IF THE DYNAMIC CONFIG IS ON, THESE ARE THE INITIAL VALUES
    AGENTS                        = 32       # Number of Agents
    PREDICTORS                    = 4        # Number of Predictors
    TRAINERS                      = 4        # Number of Trainers
    DEVICE                        = '/gpu:0' # Device
    DYNAMIC_SETTINGS              = False    # Enable the dynamic adjustment (+ waiting time to start it)
    DYNAMIC_SETTINGS_STEP_WAIT    = 20
    DYNAMIC_SETTINGS_INITIAL_WAIT = 10

    #########################################################################
    # ALGORITHM PARAMETER
    DISCOUNT                = 0.99    # Discount factor
    TIME_MAX                = 5       # Tmax
    MAX_QUEUE_SIZE          = 100     # Max size of the queue
    PREDICTION_BATCH_SIZE   = 128     # > AGENTS
    IMAGE_WIDTH             = 84      # Input of the DNN
    IMAGE_HEIGHT            = 84      # Input of the DNN
    EPISODES                = 4000000 # Total number of episodes and annealing frequency
    ANNEALING_EPISODE_COUNT = 4000000

    # OPTIMIZER PARAMETERS
    OPT_RMSPROP, OPT_ADAM   = range(2) # Initialize optimizer types as enum
    OPTIMIZER               = OPT_ADAM # Optimizer choice 
    LEARNING_RATE_START     = 1e-4     # Learning rate
    LEARNING_RATE_END       = 1e-4     # Learning rate
    RMSPROP_DECAY           = 0.99
    RMSPROP_MOMENTUM        = 0.0
    RMSPROP_EPSILON         = 0.1
    BETA_START              = 0.01     # Entropy regularization hyper-parameter
    BETA_END                = 0.01 
    BETA_ATTENTION          = 0.0      # Entropy regularization for attention modality term
    USE_GRAD_CLIP           = False    # Enable to use gradient clipping
    GRAD_CLIP_NORM          = 40.0
    LOG_EPSILON             = 1e-6     # Epsilon (regularize policy lag in GA3C)
    TRAINING_MIN_BATCH_SIZE = 40       # Training min batch size - increasing the batch size increases the stability of the algorithm, but make learning slower

    #########################################################################
    # LOG AND SAVE
    TENSORBOARD                  = True         # Enable TensorBoard
    TENSORBOARD_UPDATE_FREQUENCY = 50           # Update TensorBoard every X training steps
    SAVE_MODELS                  = True         # Enable to save models every SAVE_FREQUENCY episodes
    SAVE_FREQUENCY               = 20           # Save every SAVE_FREQUENCY episodes
    PRINT_STATS_FREQUENCY        = 1            # Print stats every PRINT_STATS_FREQUENCY episodes
    STAT_ROLLING_MEAN_WINDOW     = 1000         # The window to average stats
    RESULTS_FILENAME             = 'results.txt'# Results filename

    #########################################################################
    # MORE EXPERIMENTAL PARAMETERS 
    MIN_POLICY = 0.0 # Minimum policy
