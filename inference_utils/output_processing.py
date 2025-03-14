import huggingface_hub
import json
import numpy as np
import torch

from scipy import stats
from skimage.metrics import hausdorff_distance
from reprlib import recursive_repr
from functools import partial
from typing import Callable, Iterable, List, Set, Tuple, TypeVar, cast
from torch import Tensor, einsum
from scipy.spatial import distance, cKDTree


def check_mask_stats(img, mask, modality_type, target):
    # img: np.array, shape=(H, W, 3) RGB image with pixel values in [0, 255]
    # mask: np.array, shape=(H, W, 1) mask probability scaled to [0,255] with pixel values in [0, 255]
    # modality_type: str, see target_dist.json for the list of modality types
    # target: str, see target_dist.json for the list of targets
    
    target_dist = get_target_dist()
    
    if modality_type not in target_dist:
        raise ValueError(f"Currently support modality types: {list(target_dist.keys())}")
    
    if target not in target_dist[modality_type]:
        raise ValueError(f"Currently support targets for {modality_type}: {list(target_dist[modality_type].keys())}")
    
    ms = mask_stats(mask, img)
    
    ps = [stats.ks_1samp([ms[i]], stats.beta(param[0], param[1]).cdf).pvalue for i, param in enumerate(target_dist[modality_type][target])]
    p_value = np.prod(ps)
    
    adj_p_value = p_value**0.25    # adjustment for four test products
    
    return adj_p_value
    
    

def mask_stats(mask, img):
    # mask is a prediction mask with pixel values in [0, 255] for probability in [0, 1]
    # img is a RGB image with pixel values in [0, 255]
    if mask.max() <= 127:
        return [0, 0, 0, 0]
    return [mask[mask>=128].mean()/256, img[:,:,0][mask>=128].mean()/256, 
            img[:,:,1][mask>=128].mean()/256, img[:,:,2][mask>=128].mean()/256]
    
    
    
def combine_masks(predicts):
    # predicts: a dictionary of pixel probability, {TARGET: pred_prob}
    pixel_preds = {}
    target_area = {}
    target_probs = {}
    for target in predicts:
        pred = predicts[target]
        pred_region = np.where(pred > 0.1)
        target_area[target] = 0
        target_probs[target] = 0
        for (i,j) in zip(*pred_region):
            if (i,j) not in pixel_preds:
                pixel_preds[(i,j)] = {}
            pixel_preds[(i,j)][target] = pred[i,j]
            target_area[target] += 1
            target_probs[target] += pred[i,j]
    for target in predicts:
        if target_area[target] == 0:
            continue
        target_probs[target] /= target_area[target]
    
    # generate combined masks
    combined_areas = {t: 0 for t in predicts}
    for index in pixel_preds:
        pred_target = sorted(pixel_preds[index].keys(), key=lambda t: pixel_preds[index][t], reverse=True)[0]
        combined_areas[pred_target] += 1

    # discard targets with small areas
    discard_targets = []
    for target in predicts:
        if combined_areas[target] < 0.5 * target_area[target]:
            discard_targets.append(target)

    # keep the most confident target
    most_confident_target = sorted(predicts.keys(), key=lambda t: target_probs[t], reverse=True)[0]

    discard_targets = [t for t in discard_targets if t != most_confident_target]
    
    masks = {t: np.zeros_like(predicts[t]).astype(np.uint8) for t in predicts if t not in discard_targets}
    for index in pixel_preds:
        candidates = [t for t in pixel_preds[index] if t not in discard_targets and pixel_preds[index][t] > 0.5]
        if len(candidates) == 0:
            continue
        pred_target = max(candidates, key=lambda t: pixel_preds[index][t])
        masks[pred_target][index[0], index[1]] = 1
    
    return masks



def get_target_dist():
    huggingface_hub.hf_hub_download('microsoft/BiomedParse', filename='target_dist.json', local_dir='./inference_utils')
    huggingface_hub.hf_hub_download('microsoft/BiomedParse', filename="config.yaml", local_dir="./configs")
    return json.load(open("inference_utils/target_dist.json"))



def one_hot(t: Tensor, axis=0) -> bool:
    return sset(t, [0,1])
    # return simplex(t, axis) and sset(t, [0, 1])



# Check if there are multiple classes per pixel, axis indicating the number of classes( our case there is only one)
def simplex(t: Tensor, axis=0) -> bool:
    _sum = cast(Tensor, t.sum(axis).type(torch.float32))
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)



# Assert utils
def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())



def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)



def intersection(a: Tensor, b: Tensor) -> Tensor:
    return a * b



# Metrics
def meta_dice(sum_str: str, label: Tensor, pred: Tensor, smooth: float = 1e-8) -> Tensor:
    assert label.shape == pred.shape
    assert one_hot(label) #Check if the tensor consists of only 0-1 and that only one class is present per pixel
    assert one_hot(pred)
            
    inter_size: Tensor = einsum(sum_str, [intersection(label, pred)]).type(torch.float32)
    sum_sizes: Tensor = (einsum(sum_str, [label]) + einsum(sum_str, [pred])).type(torch.float32)

    dices: Tensor = (2 * inter_size + smooth) / (sum_sizes + smooth)

    return dices



# With this einsum subscript we take the axis ij and all 
# the rest and sum over all of them while keeping j off them 
dice_volume = partial(meta_dice, "swh->")

def meta_iou(sum_str: str, label: Tensor, pred: Tensor, smooth: float = 1e-8) -> Tensor:
    assert label.shape == pred.shape
    assert one_hot(label)
    assert one_hot(pred)

    inter_size: Tensor = einsum(sum_str, [intersection(label, pred)]).type(torch.float32) # Get only the intersection between the two image tensors
    union_size: Tensor = (einsum(sum_str, [label]) + einsum(sum_str, [pred]) - inter_size).type(torch.float32) # Get the sum of the whole volume for pred and label but make sure to level with inner untion

    ious: Tensor = (inter_size + smooth) / (union_size + smooth)

    return ious

iou_volume = partial(meta_iou, "swh->")



def max_min_distance_chunked(source, target, chunk_size):
    max_min_dist = -np.inf
    tree = cKDTree(target)  # Build KDTree for fast nearest neighbor search
    for i in range(0, len(source), chunk_size):
        chunk = source[i:i + chunk_size]
        dists, _ = tree.query(chunk)  # Find nearest distances
        max_min_dist = max(max_min_dist, np.max(dists))
    return max_min_dist

def hausdorff_distance_volume(label, pred, chunk_size=1000):
    # Get coordinates of non 0 values
    p1 = np.array(np.where(label)).T
    p2 = np.array(np.where(pred)).T

    if len(p1) == 0 or len(p2) == 0:
        return np.inf

    # Compute Hausdorff distance
    hd1 = max_min_distance_chunked(p1, p2, chunk_size) 
    hd2 = max_min_distance_chunked(p2, p1, chunk_size)
    
    return max(hd1, hd2)
