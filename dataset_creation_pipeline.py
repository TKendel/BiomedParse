
import os
import torch
import numpy as np
import matplotlib.cm as cm

from PIL import Image
from torch.nn.functional import one_hot
from matplotlib import pyplot as plt

from inference_utils.processing_utils import read_nifti_only, resize_to_original, volume_trimmer
from preprocessing import Preprocessing

'''
TODO: Watch out how the mask and images are being sliced and saved, for traing it is not needed to train on background
'''
DIR = 'data\\CT\\retrain\\img' # Path to the nifty directory
patient_numbers = [name[:6] for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]
label_dict = {0: "background", 1: "tumor", 2: "vessel", 3: "pancreas"}


for number in patient_numbers:
    if number == '100000': # not usualy needded
        img_path = f'data/CT/retrain/img/{number}_00001_0000.nii.gz'
        GT_path = f'data/CT/retrain/gt/{number}_00001.nii.gz'

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
            im = Image.fromarray(pp.img)
            im = im.convert("L")
            im.save(f"biomedparse_datasets/CTPancreas/test/{number}_{slice_iter}_CT_abdomen.png")

            # Save separate label slice
            for label in unique_labels:
                if label == 0 or not np.any(label_one_hot[: ,: , slice_iter, int(label)].numpy()): # Skip background
                    continue
                elif label == 2 or label == 3:
                    im_label = resize_to_original(label_one_hot[: ,: , slice_iter, int(label)].numpy(), w=1024, h=1024)
                    plt.imsave(f"biomedparse_datasets/CTPancreas/test_mask/{number}_{slice_iter}_CT_abdomen_{label_dict[2]}.png", im_label, cmap=cm.gray)
                elif label == 4:
                    im_label = resize_to_original(label_one_hot[: ,: , slice_iter, int(label)].numpy(), w=1024, h=1024)
                    plt.imsave(f"biomedparse_datasets/CTPancreas/test_mask/{number}_{slice_iter}_CT_abdomen_{label_dict[3]}.png", im_label, cmap=cm.gray)
                elif label == 1:
                    im_label = resize_to_original(label_one_hot[: ,: , slice_iter, int(label)].numpy(), w=1024, h=1024)
                    plt.imsave(f"biomedparse_datasets/CTPancreas/test_mask/{number}_{slice_iter}_CT_abdomen_{label_dict[1]}.png", im_label, cmap=cm.gray)

        print(f"Done with patient file {number}.")
