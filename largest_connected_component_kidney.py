import glob
import nibabel as nib
import numpy as np

from itertools import permutations
from scipy.ndimage import label, binary_erosion
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

from inference_utils.processing_utils import read_nifti_only, volume_trimmer



# import matplotlib.image as Image
# 100002, 100005, 100011, 100028, 
PDAC_patients = [100002, 100005, 100011, 100028, 100030, 100033, 100043, 100050, 100060, 100074, 100082, 100091, 100096, 100101, 100102, 100107, 100124, 100127, 100134, 100138]
for p in PDAC_patients:
    img_paths = f'results\\CT_patient_kidneys_{p}.nii.gz'
    for img_path in glob.iglob(img_paths):
        print(img_path)
        pred, nii = read_nifti_only(img_path)
        image, nii = read_nifti_only(f'data\\CT\\img\\{p}_00001_0000.nii.gz')

        print(pred.shape)

        eroded_pred = binary_erosion(pred, iterations=4).astype(pred.dtype)

        labels, num_features = label(eroded_pred) # Labels the 1's in the predictions matrix using the default cross structure(no diag connection)

        component_sizes = np.bincount(labels.ravel()) # Flatten and count occurence of each label
        component_sizes[0] = 0 # Remove backgorund

        component_sizes_filter = component_sizes[(component_sizes > 1000)]

        hi = np.percentile(component_sizes_filter, 70, axis=0)

        component_sizes_upper_95 = component_sizes_filter[component_sizes_filter > hi]

        print(component_sizes)
        print(component_sizes_filter)
        print(component_sizes_upper_95)

        # Permute all left sizes above the 95th and check if and of them when divided are almost similar, save
        kidneys = []
        for component_1,  component_2 in permutations(component_sizes_upper_95, 2):
            if component_1 / component_2 > 0.80 and component_1 / component_2 < 1:

                kidneys.append([component_1, component_2])

        # Get index position
        args = []
        for size in np.unique(np.array(kidneys)):
            args.append(np.argwhere(size == component_sizes)[0][0])

        if len(args) == 2:
            print('Found them!')
            output_mask = np.isin(labels, args).astype(np.uint8)
            final_img = nib.Nifti1Image(output_mask, nii.affine)
            nib.save(final_img, f'results\\CT_patient_kidneys_only_{p}.nii.gz')
        else:
            print("Uhoh")
            continue
