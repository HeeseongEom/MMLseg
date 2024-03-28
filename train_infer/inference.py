import os
import shutil
import tempfile

import matplotlib.pyplot as plt
from tqdm import tqdm
import monai
import nibabel as nib
from monai.data import partition_dataset
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    EnsureChannelFirstd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    EnsureTyped,
    ResizeWithPadOrCropd
)
import torch.nn.functional as F
from monai.config import print_config
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric, MeanIoU
from monai.networks.nets import SwinUNETR

from monai.data import (
    ThreadDataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
    set_track_meta,
)
import torch
import logging
import warnings
import numpy as np
logging.getLogger("monai").setLevel(logging.ERROR)  # MONAI 로거의 레벨을 'ERROR'로 설정
warnings.filterwarnings("ignore") # 경고제거용

root_dir = 'D:\\MultiModal\\research\\research1\\mymodel'
root_dir = 'D:\\Tumor_Segmentation\\severance\\models\\SwinUnetr'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available() == True:
    print('cuda is available')

val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        ScaleIntensityRanged(keys=["image"], a_min=-50, a_max=150, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 3.0),
            mode=("bilinear", "nearest"),
        ),
        EnsureTyped(keys=["image", "label"], device=device, track_meta=True),
        ResizeWithPadOrCropd(keys=["image", "label"],
        spatial_size=(320, 320, 64),
        mode='constant')
    ]
)
nib.imageglobals.logger.setLevel(40)

data_dir = "D:\\MultiModal\\research\\research1\\data\\"
data_dir = "D:\\Tumor_Segmentation\\severance\\data\\"
split_json = "dataset.json"
split_json = "dataset_1.json"
datasets = data_dir + split_json
val_files = load_decathlon_datalist(datasets, True, "validation")


val_ds = CacheDataset(
    data=val_files,
    transform=val_transforms,
    cache_num=1,
    cache_rate=1.0,
    num_workers=4
)
val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1)

set_track_meta(False)

model = SwinUNETR(
    img_size=(224, 224, 32),
    in_channels=1,
    out_channels=3,
    feature_size=48
).to(device)

for i, batch in enumerate(val_loader, start=1):
    model.load_state_dict(torch.load(os.path.join(root_dir, f"6 best_metric_model.pth")))
    model.eval()
    if i==1:
        val_inputs, val_labels = batch["image"].cuda().float(), batch["label"].cuda().float() 
        #text_path = batch["text"]
        #text_path = os.path.join(data_dir, text_path[0])
        #val_text = torch.load(text_path).to(device).float()
        #val_outputs = model(val_inputs, val_text)
        val_outputs = model(val_inputs)
        print(len(val_outputs))
        print(val_outputs[0].shape)
        val_outputs = val_outputs[0]
        import matplotlib.pyplot as plt

        val_outputs_np = val_outputs.detach().cpu().numpy()

        # 클래스 수에 따라서
        num_classes = val_outputs_np.shape[1]

        '''for i in range(num_classes):
            plt.figure(figsize=(10, 6))
            plt.hist(val_outputs_np[0, i].flatten(), bins=50, alpha=0.75, label=f'Class {i+1}')
            plt.title(f'Histogram of Output Values for Class {i+1}')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.legend()
            plt.show()'''
        val_probs = F.softmax(val_outputs, dim=1)

        val_probs_np = val_probs.detach().cpu().numpy()

        num_classes = val_probs_np.shape[1]  

        plt.figure(figsize=(20, 6))

        for i in range(num_classes):
            plt.subplot(1, num_classes, i+1)
            plt.hist(val_probs_np[0, i].flatten(), bins=50, alpha=0.75, label=f'Class {i+1}')
            plt.title(f'Histogram of Probability Values for Class {i+1}')
            plt.xlabel('Probability')
            plt.ylabel('Frequency')
            plt.legend()

        plt.tight_layout()
        plt.show()