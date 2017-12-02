import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

# sns setting
sns.set()
sns.set_style("darkgrid")

total_iter_num = 0
files = [f for f in os.listdir('.') if os.path.isfile(f)]
for file in files:
    if 'image_' in file:
        total_iter_num += 1

image_list       = []
audio_list       = []
attention_i_list = []
attention_a_list = []

for iter_num in range(total_iter_num):
    image_list.append(mpimg.imread('image_'+str(iter_num).zfill(2)+'.png'))
    audio_list.append(mpimg.imread('audio_'+str(iter_num).zfill(2)+'.png'))

    if iter_num == total_iter_num-1:
        attention_i = np.load('attention_i_'+str(iter_num).zfill(2)+'.npy')
        attention_a = np.load('attention_a_'+str(iter_num).zfill(2)+'.npy')

# Display 
plt.figure()
subplot_index = 1
for row in range(total_iter_num):
    for col in range(3):
        if col == 0:
            plt.subplot2grid((total_iter_num, 3), (row, col))
            plt.imshow(image_list[-row-1])
            plt.axis('off')

        if col == 2:
            plt.subplot2grid((total_iter_num, 3), (row, col))
            plt.imshow(audio_list[-row-1])
            plt.axis('off')

        subplot_index += 1
plt.subplot2grid((total_iter_num, 3), (0, 1), rowspan=total_iter_num)
plt.plot(attention_i, np.arange(1, len(attention_i)+1), label='Attention-image')
plt.plot(attention_a, np.arange(1, len(attention_a)+1), label='Attention-audio')
# plt.legend(loc='upper center', ncol=1, fancybox=True, shadow=True, bbox_to_anchor=(0.5, 1.15))
plt.legend(ncol=1, fancybox=True, shadow=True)
plt.title('Attention Visualization')
plt.xlabel("Probability")
plt.ylabel("Step #")
ax = plt.gca()
ax.set_xlim(-0.2, 1.2)
ax.set_yticks(np.arange(1, len(attention_i)+1))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05, hspace=0.05)
plt.show()
