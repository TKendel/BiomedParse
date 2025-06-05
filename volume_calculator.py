import numpy as np
import glob
import nibabel as nib
import os
import torch.nn.functional as F
import torch
import shutil


    
def volume_MRI(DIR):
    directories = os.listdir(DIR)

    for directory in directories:
        sub_directories = os.listdir(f"{DIR}{directory}")
        for sub_directory in sub_directories:
            files = os.listdir(f"{DIR}{directory}\\{sub_directory}")
            img = None
            for file in files:
                if 'GTV' in file:
                    img = file
                    img_paths = f'{DIR}{directory}\\{sub_directory}\\{img}'
                    for img_path in glob.iglob(img_paths):
                        input = nib.load(img_path)
                        image = input.get_fdata()
                        # if not os.path.isdir(f'DICOM_GTVs\\{directory}'):
                        #     os.mkdir(f'DICOM_GTVs\\{directory}')
                        # if not os.path.isdir(f'DICOM_GTVs\\{directory}\\{sub_directory}'):
                        #     os.mkdir(f'DICOM_GTVs\\{directory}\\{sub_directory}')

                        # shutil.copy(f'{DIR}{directory}\\{sub_directory}\\{img}', f'DICOM_GTVs\\{directory}\\{sub_directory}\\{img}')

                        voxel_count = np.count_nonzero(image)
                        sx, sy, sz = input.header.get_zooms()
                        volume = sx * sy * sz * voxel_count
                        spatial_unit, _ = input.header.get_xyzt_units()

                        with open("volumes_MRI.txt", "a") as f:
                            f.write(f"Patient {directory} at scan {sub_directory} has a GTV of volume: {volume:.2f} {spatial_unit}³\n")

        with open("volumes_MRI.txt", "a") as f:
                            f.write(f"\n")


def volume_CT(DIR):
    files = os.listdir(DIR)
    for file in files:
        img_path = f'{DIR}/{file}'
        input = nib.load(img_path)
        labels = input.get_fdata()

        label_one_hot = F.one_hot(torch.tensor(labels).long(), num_classes=-1)

        if 1 not in np.unique(labels):
            continue
        else:
            voxel_count = np.count_nonzero(label_one_hot[:,:,:,1])
            sx, sy, sz = input.header.get_zooms()
            volume = sx * sy * sz * voxel_count
            spatial_unit, _ = input.header.get_xyzt_units()

            with open("volumes_CT.txt", "a") as f:
                f.write(f"Patient {file} has a GTV of volume: {volume:.2f} {spatial_unit}³\n")

    with open("volumes_CT.txt", "a") as f:
                        f.write(f"\n")





# DIR = 'data\AUMC\\'
DIR = 'data/CT/gt/'

volume_CT(DIR)