import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

raw = []

path = 'thesis_plots/images/'
for subdir, dirs, files in os.walk(path):
    for file in files:
        im = plt.imread(path+file)
        raw.append(im)


# fig, ax = plt.subplots()
# ax.imshow(raw[1])
# ax.axis('off')
# plt.subplots_adjust(wspace=0, hspace=0)

# plt.show()

nrow = 3
ncol = 3

fig = plt.figure( figsize=(8, 8)) 
fig.subplots_adjust(0,0,1,1)

gs = gridspec.GridSpec(nrow, ncol,
         wspace=0.0, hspace=0.0, 
         top=1.-0.5/(nrow+1), bottom=0.5/(nrow+1), 
         left=0.5/(ncol+1), right=1-0.5/(ncol+1)) 



n = 8
for i in range(nrow):
    for j in range(ncol):
        ax = plt.subplot(gs[j])
        if n ==8:
            ax.set_title("Raw Image")
        if n == 7:
            ax.set_title("Ground Truth")
        if n== 6:
            ax.set_title("3D Representation")
        ax.imshow(raw[n])
        ax.axis('off')
        n-=1
# ax[0,0].set_title("Raw Image")
# ax[0,1].set_title("Ground Truth")
# ax[0,2].set_title("3D")
# plt.show()
plt.savefig(f"thesis_plots/problem_case.png", bbox_inches='tight')