import os
import numpy as np
import nibabel as nib
import cv2
import matplotlib.pyplot as plt
from preprocessing import Preprocessing

from scipy.ndimage import center_of_mass
from scipy.interpolate import RegularGridInterpolator

from inference_utils.processing_utils import read_nifti_only, volume_trimmer

def interval_mapping(image, from_min, from_max, to_min, to_max):
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = np.array((image - from_min) / float(from_range), dtype=float)
    return to_min + (scaled * to_range)

# function to make volume isotropic with voxel size 1x1x1
def do_interpolate(image_data, steps):
    dx, dy, dz = 0.6796875, 0.6796875, 5  # step sizes
    x, y, z = [steps[k] * np.arange(image_data.shape[k]) for k in range(len(steps))] 
    f_scan = RegularGridInterpolator((x, y, z), image_data, method='linear') 
    new_grid = np.mgrid[0:x[-1]:dx, 0:y[-1]:dy, 0:z[-1]:dz]
    new_grid = np.moveaxis(new_grid, (0, 1, 2, 3), (3, 0, 1, 2)) 
    return f_scan(new_grid)

def createMIP_transverse(np_img, slices_num):
    ''' create the mip image from original image, slice_num is the number of slices for maximum intensity projection'''
    img_shape = np.shape(np_img)
    np_mip = np.zeros_like(np_img)
    for i in range(img_shape[2]-slices_num):
        np_mip[:, :, i] = np.amax(np_img[:, :, i:(i + slices_num)], axis=2)
    return np_mip


def plot_histogram(GT, label):
    plt.hist(GT.flatten(), bins=100, alpha=0.5, label=label)

PDAC_patients = [100002, 100005, 100011, 100030, 100033, 100043, 100050, 100060, 100074, 100082, 100091, 100096, 100101, 100102, 100124, 100127, 100134]

PDAC_patients = [100082]
idx = 350
# works: 33, 11 , 30, 134, 60
#kinda works: 50, 102
# does not works: 5, 2 28,43,74,82,91,96,101,124
# path = 'data\\CT\\img\\'
# for subdir, dirs, files in os.walk(path):
#     for file in files:
#         for p in PDAC_patients:
#             if str(p) in file:
#                 pred, nii = read_nifti_only(path+file)
#                 ground_img, _ = read_nifti_only("examples/amos_0328.nii.gz")
#                 isoscale_resolution = nii.header['pixdim'][1:4]
#                 isoscale_resolution_gt = _.header['pixdim'][1:4]
#                 print(isoscale_resolution)
#                 print(isoscale_resolution_gt)
#                 # process image with intensity range 0.5-99.5 percentile
#                 # lower_bound, upper_bound = np.percentile(pred[pred > -1200], 0.1), np.percentile(pred, 99.5)
                
#                 # image_data_pre = np.clip(pred, lower_bound, upper_bound)

#                 # plot_histogram(ground_img, "examples/amos_0328.nii.gz")
#                 # plot_histogram(pred, file)
#                 # plt.legend()
#                 # plt.show()
#             # aa = do_interpolate(pred, isoscale_resolution)

#         # proj = interval_mapping(pred, -150, 250, 0, 1)

        
#         # img = cv2.imread(path+file)

#                 pp = Preprocessing(pred[:,:,idx], path+file)
#                 pp.normalize()
#         # pp.gammaCorrection(0.5)
#         # pp.adaptiveEqualize()
#                 pp.CLAHEClipping()
#         # pp.bilateralFilter()
#         # pp.equalize()
#         # pp.bilateralFilter()

#         # pp.logCorrection(1.1)

#         # pp.histogramMatching(ground_img[68])

#                 pred[:,:, idx] = pp.img
        
#         # pp.save()
#             # image_data_pre = np.rot90(image_data_pre, 2)

#                 final_img = nib.Nifti1Image(pred, nii.affine)
#                 nib.save(final_img, f'test2.nii.gz')


"""
Rotation does nothing, clipping the histogram values so it matches the atoms data does also not help, resizing voxel size also does not help

CLAHE helps on some scans like 28,5 60 but not on scans like 43 74
"""


# DIR = 'data\AUMC\\001\\001-0\image.nii.gz'
# pred, nii = read_nifti_only(DIR)

# imgplot = plt.imshow(pred[:,:, 40],cmap='gray')
# print()
# plt.show()


