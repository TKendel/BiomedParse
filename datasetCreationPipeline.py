
import os
import torch
import numpy as np
import matplotlib.cm as cm

from PIL import Image
from torch.nn.functional import one_hot
from matplotlib import pyplot as plt

from inference_utils.processing_utils import read_nifti_only, resize_to_original, volume_trimmer


DIR = 'data/CT/retrain/gt/'
patient_numbers = [name[:6] for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]
label_dict = {0: "background", 1: "pancreatic+tumor", 2: "pancreatic+veins", 3: "pancreatic+arteries", 4: "pancreas+parenchyma", 5: "pancreatic+duct", 6: "bile+duct"}

for number in patient_numbers:
    img_path = f'data/CT/retrain/img/{number}_00001_0000.nii.gz'
    GT_path = f'data/CT/retrain/gt/{number}_00001.nii.gz'

    # The result predictions are soft, hence why he thresholds 0.5 to set make them 0-1
    image, nii = read_nifti_only(img_path)
    GT, nii = read_nifti_only(GT_path)

    assert image.shape == GT.shape

    trimmed_GT = volume_trimmer(GT)

    if type(trimmed_GT) is None :
        print(f"No label found for patient{number}")
        continue        

    unique_labels = np.unique(trimmed_GT)
    # One hot encode mask labels
    label_one_hot = one_hot(torch.tensor(trimmed_GT).long(), num_classes=-1)

    for slice_iter in range(44, 45):
        # Save img slice
        im = resize_to_original(image[: ,: , slice_iter], w=1024, h=1024)
        im = Image.fromarray(im)
        im = im.convert("L")
        im.save(f"biomedparse_datasets/CTPancreas/train/{number}{slice_iter}_CT_abdomen.png")

        """
        TODO: this would be used to just have one mask of all the labels, however need to research if this would help or do the oposite
        The mode can take multi modal masks but not sure if this is helping the model and if separate is better
        """
        # # Save label slice
        # im_label = resize_to_original(GT[:, :, slice_iter], w=1024, h=1024)
        # im_label = Image.fromarray((im_label * 255).astype(np.uint8))
        # print(im_label)
        # im_label = im_label.convert("L")
        # im_label.save(f"biomedparse_datasets/CTPancreas/train_mask/{number}{slice_iter}_CT_abdomen_test.png")

        # Save separate label slice
        for label in label_dict.keys():
            if label == 0 or label not in unique_labels or not np.any(label_one_hot[: ,: , slice_iter, label].numpy()): # Skip background
                continue
            else:
                im_label = resize_to_original(label_one_hot[: ,: , slice_iter, label].numpy(), w=1024, h=1024)
                plt.imsave(f"biomedparse_datasets/CTPancreas/train_mask/{number}{slice_iter}_CT_abdomen_{label_dict[label]}.png", im_label, cmap=cm.gray)
