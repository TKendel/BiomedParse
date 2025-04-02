import numpy as np
import torch
import torch.nn.functional as F


from inference_utils.processing_utils import read_nifti_only, process_intensity_image, resize_to_original
from inference_utils.output_processing import dice_volume



import numpy as np

def dice_coefficient(ref_segment, out_segment):
    # Ensure inputs are binary (0s and 1s)
#     ref_segment = ref_segment.dtype(bool)
#     out_segment = out_segment.dtype(bool)
    
    # Compute the intersection
    common = np.logical_and(ref_segment, out_segment)
    
    # Compute Dice coefficient
    a = torch.sum(common)   # Intersection count
    b = torch.sum(torch.tensor(ref_segment))  # Total pixels in reference
    c = torch.sum(out_segment)  # Total pixels in output

    # Avoid division by zero
    return 2 * a / (b + c) if (b + c) > 0 else 1.0

def compute_dice_coefficient(mask_gt, mask_pred):
  """Computes soerensen-dice coefficient.

  compute the soerensen-dice coefficient between the ground truth mask `mask_gt`
  and the predicted mask `mask_pred`.

  Args:
    mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
    mask_pred: 3-dim Numpy array of type bool. The predicted mask.

  Returns:
    the dice coeffcient as float. If both masks are empty, the result is NaN.
  """
  volume_sum = mask_gt.sum() + mask_pred.sum()
  if volume_sum == 0:
    return np.nan
  volume_intersect = (mask_gt & mask_pred).sum()
  return 2*volume_intersect / volume_sum


# For single file test
patients = ['100005']

for patient in patients:

     path_label = f'data//CT//gt//{patient}_00001.nii.gz'
     label, nii = read_nifti_only(path_label)

     # One hot encode multiple classes
     label_one_hot = F.one_hot(torch.tensor(label).long(), num_classes=-1)
     # print(np.unique(label_one_hot[:,:,:,1]))
     # print("Unique values currently are :", torch.unique(label_one_hot))

     path_pred = f'results//CT_patient_{patient}.nii.gz'
     pred, nii = read_nifti_only(path_pred)

     pred = torch.tensor(pred)
     # Threshold soft predictions to hard, same as the writer
     pred = (pred > 0.5).float()


     # print(label_one_hot.shape)
     # print(np.unique(label_one_hot[:,:,:,3]))
     # print(pred.shape)
     # print(np.unique(pred))

     rez = dice_coefficient(label_one_hot[:,:,:,1], pred)
     print(rez)

     dice_old = dice_volume(torch.permute(label_one_hot[:,:,:,1], (2, 0, 1)), torch.permute(pred, (2, 0, 1)))
     print(dice_old)

     print(label_one_hot[:,:,:,1].dtype)
     print(pred.dtype)
     rez2 = compute_dice_coefficient(np.array(label_one_hot[:,:,:,1]), np.array(pred).astype(int))

     print(rez2)



'''
def meta_dice(sum_str: str, label: Tensor, pred: Tensor, smooth: float = 1e-8) -> Tensor:
    assert label.shape == pred.shape
    assert one_hot(label) #Check if the tensor consists of only 0-1 and that only one class is present per pixel
    assert one_hot(pred)
            
    inter_size: Tensor = einsum(sum_str, [intersection(label, pred)]).type(torch.float32)
    sum_sizes: Tensor = (einsum(sum_str, [label]) + einsum(sum_str, [pred])).type(torch.float32)

    dices: Tensor = (2 * inter_size + smooth) / (sum_sizes + smooth)

    return dices

'''
