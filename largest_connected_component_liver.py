import glob
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from scipy.ndimage import label, binary_closing, binary_erosion, center_of_mass

from inference_utils.processing_utils import read_nifti_only, volume_trimmer

PDAC_patients = [100002, 100005, 100011, 100030, 100033, 100043, 100050, 100060, 100074, 100082, 100091, 100096, 100101, 100102, 100124, 100127, 100134]
# PDAC_patients = [100091,100107,100138]

for p in PDAC_patients:
    img_paths = f'results\\CT_patient_kidney_{p}.nii.gz'
    results = []
    for img_path in glob.iglob(img_paths):
        pred, nii = read_nifti_only(img_path)
        pred = (pred > 0.5).astype(np.uint8)
        image, nii = read_nifti_only(f'data\\CT\\img\\{p}_00001_0000.nii.gz')

        print(img_path)

        eroded_pred = binary_closing(pred, iterations=1).astype(pred.dtype)

        labels, features = label(eroded_pred)
        segmentation_sizes = np.bincount(labels.ravel())
        segmentation_sizes[0] = 0

        largest_component = segmentation_sizes.argmax()
        largest_segmentation = (labels == largest_component).astype(np.uint8)

        first, last = volume_trimmer(largest_segmentation)
        middle = int((last+first)/2)
        com = int(center_of_mass(largest_segmentation)[2])

        first_half = np.sum(largest_segmentation[ :, :, :com])
        second_half = np.sum(largest_segmentation[:, :, com:])

        # Check liver density halfs
        if first_half > second_half:
            results.append(1)
        else:
            results.append(-1)

        # Check center of mass relation towards middle
        if com > image.shape[2]/2:
            results.append(1)
        else:
            results.append(-1)

        # Check postions of biggest slice 
        slice_areas = largest_segmentation.sum(axis=(0, 1))
        max_idx = np.argmax(slice_areas)
        z_ratio = max_idx / largest_segmentation.shape[2] 
        
        if z_ratio < 0.5:
            results.append(1)
        else:
            results.append(-1)


        buffer_1 = int(image.shape[2] * 0.08)
        buffer_2 = int(image.shape[2] * 0.07)
        final_img = nib.Nifti1Image(pred, nii.affine)
        nib.save(final_img, f'results\\CT_patient_pancreas_tumor_final_{p}.nii.gz')

        if sum(results) < 0:
            print("not flipped")
            image = image[:,:, first-buffer_2:last-buffer_1]
            final_img = nib.Nifti1Image(image, nii.affine)
            nib.save(final_img, f'results\\CT_patient_liver_only_img{p}.nii.gz')
        else:
            print("flipped")
            image = image[:,:, first+buffer_1:last+buffer_2]
            final_img = nib.Nifti1Image(image, nii.affine)
            nib.save(final_img, f'results\\CT_patient_liver_only_img{p}.nii.gz')
