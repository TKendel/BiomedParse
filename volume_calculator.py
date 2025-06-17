import numpy as np
import glob
import nibabel as nib
import os
import pydicom
import torch.nn.functional as F
import torch
import shutil
import time


    
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

        if int(file[:6]) >= 100780:
            print(file)
            img_path = f'{DIR}/{file}'
            input = nib.load(img_path)
            labels = torch.tensor(input.get_fdata()).long()
            # labels = labels.to(device='cuda')
            unique_labels = torch.unique(labels)

            if 1 not in unique_labels:
                print('No GTV')
                continue
            else:
                label_one_hot = F.one_hot(labels, num_classes=-1)
                # label_one = F.one_hot(labels, num_classes=2)
                # print(label_one_hot[:,:,:,1].shape)
                # print(label_one[:,:,:,0].shape)
                # new_image1 = nib.Nifti1Image(np.array(label_one_hot[:,:,:,1]), affine=input.affine)
                # new_image2 = nib.Nifti1Image(np.array(label_one[:,:,:,0]), affine=input.affine)
                # nib.save(new_image1, 'a.nii.gz')
                # nib.save(new_image2, 'b.nii.gz')
                voxel_count = torch.count_nonzero(label_one_hot[:,:,:,1])
                sx, sy, sz = input.header.get_zooms()
                volume = sx * sy * sz * voxel_count
                spatial_unit, _ = input.header.get_xyzt_units()

                with open("volumes_CT_3.txt", "a") as f:
                    f.write(f"Patient {file} has a GTV of volume: {volume:.2f} {spatial_unit}³\n")
                print(file, "Done")
            torch.cuda.empty_cache()
            time.sleep(2)


    with open("volumes_CT_3.txt", "a") as f:
                        f.write(f"\n")

DIR = 'data\AUMC\\'
