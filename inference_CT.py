import numpy as np
import glob
import torch
import json
import torch.nn.functional as F
import nibabel as nib
import huggingface_hub


from matplotlib import pyplot as plt
from PIL import Image

from modeling.BaseModel import BaseModel
from modeling import build_model
from utilities.distributed import init_distributed
from utilities.arguments import load_opt_from_config_files
from utilities.constants import BIOMED_CLASSES


from inference_utils.inference import interactive_infer_image
from inference_utils.output_processing import dice_volume, iou_volume, hausdorff_distance_volume
from inference_utils.processing_utils import read_nifti_only, process_intensity_image, resize_to_original, volume_trimmer



with open('tokens.json') as f:
    tokens = json.load(f)

HF_TOKEN = tokens['hugging_face']
huggingface_hub.login(HF_TOKEN)

opt = load_opt_from_config_files(["configs/biomedparse_inference.yaml"])
opt = init_distributed(opt)

# Load model from pretrained weights
pretrained_pth = 'pretrained/biomed_parse.pt'
pretrained_pth = 'hf_hub:microsoft/BiomedParse'

model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
with torch.no_grad():
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(BIOMED_CLASSES + ["background"], is_eval=True)
