
import os
import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt
import json
import torch
import glob
import matplotlib.cm as cm

from PIL import Image, ImageOps
from matplotlib import pyplot as plt
from pydicom import dcmread
from torch.nn.functional import one_hot

from inference_utils.output_processing import dice_volume, iou_volume, hausdorff_distance_volume
from inference_utils.processing_utils import read_nifti_only, resize_to_original, volume_trimmer



"TODO: CITE THESE HELPFUL BUNCH"
from dcmrtstruct2nii import dcmrtstruct2nii, list_rt_structs



def dicom_reader():

    DIR = 'Dicom_clinc'

    sex = []
    directories = os.listdir(DIR)
    done = 0
    for directory in directories:
        if done == 1:
            done = 0 
            continue
        sub_directories = os.listdir(f"{DIR}\\{directory}")
        for sub_directory in sub_directories:
            files = os.listdir(f"{DIR}\\{directory}\\{sub_directory}")
            rtss = None
            if done == 1:
                done = 0
                break
            for file in files:
                if file[:5] == "image":
                    done = 1
                    ds = dcmread(f'Dicom_clinc\\{directory}\\{sub_directory}\\{file}')
                    sex.append(ds[(0x0010, 0x0040)].value)
                    break


def json_parser():
    f = open('biomedparse_v3_eval_results_abdomen+tumor.json')
    scores = json.load(f)

    for datatype in scores:
        print(f" 1 {datatype}")
        for evaltype in scores[datatype]:
            print(f" 2 {evaltype}")

            if 'instance_results' in scores[datatype][evaltype]:
                scores = scores[datatype][evaltype]['scores']['mDice']
                # scores[datatype][evaltype]['scores'] = scores[datatype][evaltype]['scores']['mDice']

    print(scores)

def dataset_creation_pipeline(DIR):

    directories = os.listdir(DIR)
    PDAC_patients = ['001', '012', '016', '021', '046', '047', '061', '069']

    for directory in directories:
        if directory not in PDAC_patients and int(directory) > 25:
            sub_directories = os.listdir(f"{DIR}{directory}")
            for sub_directory in sub_directories:
                files = os.listdir(f"{DIR}{directory}\\{sub_directory}")
                raw = None
                label = None
                for file in files:
                    if 'image' in file:
                        raw = file
                    if 'GTV' in file:
                        label = file
                
                if not raw or not label:
                    continue
                else:
                    image, nii = read_nifti_only(f"{DIR}{directory}\\{sub_directory}\\{raw}")
                    GT, nii = read_nifti_only(f"{DIR}{directory}\\{sub_directory}\\{label}")

                    assert image.shape == GT.shape

                    first, last = volume_trimmer(GT)

                    print(first,last)

                    if first and last == 0 :
                        print(f"No label found for patient{directory}\\{sub_directory}\\{raw}")
                        continue     

                    # Buffer space to capture the possible PDAC
                    buffer_1 = int(image.shape[2] * 0.08)
                    buffer_2 = int(image.shape[2] * 0.07)

                    print(first-buffer_1 ,last+buffer_2)
                    trimmed_GT = GT[:, :, first-buffer_1 : last+buffer_2]
                    trimmed_image = image[:, :, first-buffer_1 : last+buffer_2] 

                    unique_labels = np.unique(trimmed_GT)

                    for slice_iter in range(trimmed_image.shape[2]):
                        # Save img slice
                        "TODO: rename function to something more generic to avoid confusion"
                        im = resize_to_original(trimmed_image[: ,: , slice_iter], w=1024, h=1024)
                        im = Image.fromarray(im)
                        im = ImageOps.grayscale(im)
                        # im = im.convert("RGB")
                        im.save(f"biomedparse_datasets\AUMC\\train\\{sub_directory}_{slice_iter}_MRI_abdomen.png")

                        # Save separate label slice
                        print(trimmed_GT.shape)
                        for label in unique_labels:
                            if not np.any(trimmed_GT[: ,: , slice_iter]): # Skip background
                                im_label = resize_to_original(trimmed_GT[: ,: , slice_iter], w=1024, h=1024)
                                plt.imsave(f"biomedparse_datasets\AUMC\\train_mask\\{sub_directory}_{slice_iter}_MRI_abdomen_background.png", im_label, cmap=cm.gray)
                            else:
                                im_label = resize_to_original(trimmed_GT[: ,: , slice_iter], w=1024, h=1024)
                                plt.imsave(f"biomedparse_datasets\AUMC\\train_mask\\{sub_directory}_{slice_iter}_MRI_abdomen_tumor.png", im_label, cmap=cm.gray)

                    print(f"Done with patient file {directory}\\{sub_directory}\\{raw}.")


def create_dataset_PANORAMA(DIR):
    patient_numbers = [name[:6] for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]
    label_dict = {0: "background", 1: "tumor", 2: "vessel", 3: "pancreas"}

    for number in patient_numbers:
        if number == '100598': # not usualy needded
            img_path = f'data\CT\\test\{number}_00001_0000.nii.gz'
            GT_path = f'data\CT\\test_mask\{number}_00001.nii.gz'

            image, nii = read_nifti_only(img_path)
            GT, nii = read_nifti_only(GT_path)

            assert image.shape == GT.shape

            first, last = volume_trimmer(GT)

            if first and last == 0 :
                print(f"No label found for patient{number}")
                continue     

            trimmed_GT = GT[:, :, first:last]
            trimmed_image = image[:, :, first:last]    

            unique_labels = np.unique(trimmed_GT)
            # One hot encode mask labels
            label_one_hot = one_hot(torch.tensor(trimmed_GT).long(), num_classes=-1)

            for slice_iter in range(trimmed_image.shape[2]):
                # Save img slice
                "TODO: rename function to something more generic to avoid confusion"
                im = resize_to_original(trimmed_image[: ,: , slice_iter], w=1024, h=1024)
                # pp = Preprocessing(trimmed_image[: ,: , slice_iter], 'test')
                # pp.normalize()
                # pp.CLAHEClipping()
                im = Image.fromarray(im)
                im = im.convert("L")
                im.save(f"data\CT\cut_raw\{number}_{slice_iter}_CT_abdomen.png")

                # Save separate label slice
                for label in unique_labels:
                    if label == 0 or not np.any(label_one_hot[: ,: , slice_iter, int(label)].numpy()): # Skip background
                        continue
                    elif label == 2 or label == 3:
                        im_label = resize_to_original(label_one_hot[: ,: , slice_iter, int(label)].numpy(), w=1024, h=1024)
                        plt.imsave(f"data/CT/cut_gt/{number}-{slice_iter}_CT_abdomen_{label_dict[2]}.png", im_label, cmap=cm.gray)
                    elif label == 4:
                        im_label = resize_to_original(label_one_hot[: ,: , slice_iter, int(label)].numpy(), w=1024, h=1024)
                        plt.imsave(f"data/CT/cut_gt/{number}-{slice_iter}_CT_abdomen_{label_dict[3]}.png", im_label, cmap=cm.gray)
                    elif label == 1:
                        im_label = resize_to_original(label_one_hot[: ,: , slice_iter, int(label)].numpy(), w=1024, h=1024)
                        plt.imsave(f"data/CT/cut_gt/{number}-{slice_iter}_CT_abdomen_{label_dict[1]}.png", im_label, cmap=cm.gray)
            print(f"Done with patient file {number}.")


def eval_MRI(DIR):
    # directories = os.listdir(DIR)

    PDAC_patients = ['001', '012', '016', '021', '046', '047', '061', '069']

    for directory in PDAC_patients:
        sub_directories = os.listdir(f"{DIR}{directory}")
        for sub_directory in sub_directories:
            files = os.listdir(f"{DIR}{directory}\\{sub_directory}")
            img = None
            for file in files:
                if 'mask_GTV' in file:
                    img = file
                    img_paths = f'{DIR}{directory}\\{sub_directory}\\{img}'

                    for img_path in glob.iglob(img_paths):

                        # path_label = f'data//CT//gt//{patient}_00001.nii.gz'
                        gt, nii = read_nifti_only(img_path)

                        gt = gt.astype(float) / 255
                        gt = torch.tensor(gt)

                        # One hot encode multiple classes
                        unique_labels = np.unique(gt)

                        first, last = volume_trimmer(gt)
                        gt = gt[:,:,first:last]

                        path_pred = f'results_pub\\AUMC\\MRI_patient_{sub_directory}_base.nii.gz'
                        pred, nii = read_nifti_only(path_pred)

                        pred = pred[:, :, first:last]
                        pred = torch.tensor(pred)

                        # 1 is currently PDAC lessions, watch out for which label we are calculating
                        if 1 not in unique_labels:
                            continue
                        else:
                            dice = dice_volume(torch.permute(gt, (2, 0, 1)), torch.permute(pred, (2, 0, 1)))
                            iou = iou_volume(torch.permute(gt, (2, 0, 1)), torch.permute(pred, (2, 0, 1)))
                            hausdorff = hausdorff_distance_volume(torch.permute(gt, (2, 0, 1)), torch.permute(pred, (2, 0, 1)))

                            with open("metrics_MRI_base.txt", "a") as f:
                                f.write(f"3D_DICE score for patient {sub_directory} is : {dice}\n")
                                f.write(f"3D_IoU score for patient {sub_directory} is : {iou}\n")
                                f.write(f"3D_HD score for patient {sub_directory} is : {hausdorff}\n")
                                f.write("\n")

                        print(f"Patient {sub_directory} done!")

def eval_CT():
    PDAC_patients = [100002, 100005, 100011, 100030, 100033, 100043, 100050, 100060, 100074, 100082, 100091, 100096, 100101, 100102, 100124, 100127, 100134]

    for patient in PDAC_patients:

        path_label = f'data//CT//gt//{patient}_00001.nii.gz'
        label, nii = read_nifti_only(path_label)

        # One hot encode multiple classes
        unique_labels = np.unique(label)

        buffer = int(label.shape[2] * 0.05)

        label_one_hot = F.one_hot(torch.tensor(label).long(), num_classes=-1)

        label_one_hot = label_one_hot[:, :, :, 1]
        first, last = volume_trimmer(label_one_hot)
        label_one_hot = label_one_hot[:,:,first:last]

        path_pred = f'results_pub//CT_patient_{patient}_LR=-5,b=2,fullv2.nii.gz'
        pred, nii = read_nifti_only(path_pred)

        pred = pred[:, :, first:last]
        pred = torch.tensor(pred)


        # 1 is currently PDAC lessions, watch out for which label we are calculating
        if 1 not in unique_labels:
            continue
        else:
            dice = dice_volume(torch.permute(label_one_hot, (2, 0, 1)), torch.permute(pred, (2, 0, 1)))
            iou = iou_volume(torch.permute(label_one_hot, (2, 0, 1)), torch.permute(pred, (2, 0, 1)))
            hausdorff = hausdorff_distance_volume(torch.permute(label_one_hot, (2, 0, 1)), torch.permute(pred, (2, 0, 1)))

            with open("metrics_full_dataset_v2.txt", "a") as f:
                f.write(f"3D_DICE score for patient {patient} is : {dice}\n")
                f.write(f"3D_IoU score for patient {patient} is : {iou}\n")
                f.write(f"3D_HD score for patient {patient} is : {hausdorff}\n")
                f.write("\n")

        print(f"Patient {patient} done!")