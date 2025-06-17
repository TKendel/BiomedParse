import os
import re 
import pydicom
import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd



def grid_plot():
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


def dist_plot(path):
    volumes = []
    with open(path) as f:
        for line in f.readlines():
            if line.strip():
                pattern = '(?<=:).+(\d*\.?\d+)'
                output = re.search(pattern, line)
                volumes.append(float(output.group()))            
    df = pd.DataFrame({"GTV_Volume": volumes})

    ax = sbn.histplot(x='GTV_Volume', data=df, palette='muted', legend=False, bins=150 )
    ax.set(xlabel = 'Volume mm³')
    ax.set_title('GTV Volume Distribution')
    plt.savefig(f"MRI_GTV_Volume.png", bbox_inches='tight')
    # plt.show()

# 18 PATIENTS ARE MISSING AGE
def age_DICOM_plot(DIR):
    directories = os.listdir(DIR)
    sex = []
    i = 0
    for directory in directories:
        sub_directories = os.listdir(f"{DIR}{directory}")
        for sub_directory in sub_directories:
            files = os.listdir(f"{DIR}{directory}\\{sub_directory}")

            img_path = f'{DIR}{directory}\\{sub_directory}\\{files[2]}'
            dicom_meta = pydicom.filereader.dcmread(img_path)
            if dicom_meta[0x10,0x40].value == 'M' or dicom_meta[0x10,0x40].value == 'F':
                sex.append(dicom_meta[0x10,0x40].value)
            else:
                i+=1

    df = pd.DataFrame({"Sex": sex})
    print("No sex provided for ",i, "patientes.")
    ax = sbn.histplot(x='Sex', data=df, palette='muted', legend=False)
    ax.set(xlabel = 'Patient Sex (M-Male/F-Female')
    ax.set_title('Sex Distribution')
    plt.savefig(f"MRI_sex.png", bbox_inches='tight')


dist_plot('volumes.txt')
# age_DICOM_plot('Dicom_clinc/')