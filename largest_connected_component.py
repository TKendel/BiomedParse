import glob
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import ndimage
import nibabel as nib
from inference_utils.processing_utils import read_nifti_only, volume_trimmer
# import matplotlib.image as Image

PDAC_patients = [100002, 100005, 100011, 100028, 100030, 100033, 100043, 100050, 100060, 100074, 100082, 100091, 100096, 100101, 100102, 100107, 100124, 100127, 100134, 100138]
for p in PDAC_patients:
    img_paths = f'results_pub\\CT_patient_liver_{p}.nii.gz'
    for img_path in glob.iglob(img_paths):
        pred, nii = read_nifti_only(img_path)
        image, nii = read_nifti_only(f'data\\CT\\img\\{p}_00001_0000.nii.gz')

        print(pred.shape)

        labels, features = ndimage.label(pred)
        segmentation_sizes = np.bincount(labels.ravel())
        segmentation_sizes[0] = 0

        largest_component = segmentation_sizes.argmax()

        largest_segmentation = (labels == largest_component).astype(np.uint8)

        first, last = volume_trimmer(largest_segmentation)

        buffer = int(image.shape[2] * 0.1)
        image = image[:,:, first+buffer:last+buffer]

        imgplot = plt.imshow(image[:,:,0])
        plt.show()
        imgplot = plt.imshow(image[:,:,-1])
        plt.show()
        exit()
        final_img = nib.Nifti1Image(largest_segmentation, nii.affine)
        nib.save(final_img, f'results_pub\\CT_patient_liver_only_{p}.nii.gz')
