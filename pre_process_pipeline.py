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
    dx, dy, dz = 1, 1, 1   # step sizes
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


path = 'biomedparse_datasets\\BiomedParseData-Demo\\demo\\'
for subdir, dirs, files in os.walk(path):
    for file in files:        
        # pred, nii = read_nifti_only(path+file)
        # ground_img, _ = read_nifti_only("examples/amos_0328.nii.gz")
        # isoscale_resolution = nii_.header['pixdim'][1:4]
        # aa = do_interpolate(pred, isoscale_resolution)

        # proj = interval_mapping(pred, -150, 250, 0, 1)

        
        img = cv2.imread(path+file)

        pp = Preprocessing(img, path+file)
        pp.normalize()
        # pp.gammaCorrection(0.5)
        # pp.adaptiveEqualize()
        pp.CLAHEClipping()
        # pp.bilateralFilter()
        # pp.equalize()
        # pp.bilateralFilter()

        # pp.logCorrection(1.1)

        # pp.histogramMatching(ground_img[68])

        # pred[:,:, 450] = pp.img
        
        pp.save()

        # final_img = nib.Nifti1Image(pred, nii.affine)
        # nib.save(final_img, f'test2.nii.gz')



