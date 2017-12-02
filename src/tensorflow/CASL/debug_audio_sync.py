import os, glob, cv2, argparse
import numpy as np
from scipy.io.wavfile import read, write


# Argument
parser = argparse.ArgumentParser()
parser.add_argument('-pid','--pid', help='Enter pid to debug audio sync', required=True)
args = vars(parser.parse_args())

# Setup pid
pid = args['pid']
os.chdir('debug/'+pid)# Change directory

# Concatenation of audio
audio_concat = []
for item in sorted(glob.glob("*.wav")):
	freq, audio = read(item)

	for data in audio:
		audio_concat.append(data)

# Save 
freq = 30720
write('sound.wav', freq, np.asarray(audio_concat))
