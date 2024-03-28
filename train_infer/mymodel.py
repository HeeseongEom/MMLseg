import os
import shutil
import tempfile

import matplotlib.pyplot as plt
from tqdm import tqdm

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
logging.getLogger("monai").setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("ignore")
#print_config()
import numpy as np


root_dir = 'D:\\MultiModal\\research\\research1\\mymodel'
print(root_dir)
rand_num = np.random.randint(10)
print(rand_num)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available() == True:
    print('cuda is available')

import monai
print(monai.__version__)
###------------------------------------------Transforms------------------------------------------
num_samples = 1

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        
        ScaleIntensityRanged(
            keys=["image"],
            #a_min=-175,
            #a_max=250,
            #실험적으로 수정
            #a_min=-100,
            #a_max=200,
            a_min=-50,
            a_max=200,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            #slice_thickness=3
            pixdim=(1.0, 1.0, 3.0),
            mode=("bilinear", "nearest"),
        ),

        EnsureTyped(keys=["image", "label"], device=device, track_meta=False),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.50,
        ),
        #RandCropByPosNegLabeld에서 smaller=True하면 이것도 해야함, spatial_size divisible by 32주의
        ResizeWithPadOrCropd(keys=["image", "label"],
        spatial_size=(320, 320, 64),
        mode='constant')
    ]        
    
)

val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        ScaleIntensityRanged(keys=["image"], a_min=-50, a_max=200, b_min=0.0, b_max=1.0, clip=True),
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


import nibabel as nib
from monai.data import partition_dataset
nib.imageglobals.logger.setLevel(40)

data_dir = "D:\\MultiModal\\research\\research1\\data\\"
split_json = "dataset.json"

datasets = data_dir + split_json

datalist = load_decathlon_datalist(datasets, True, "training")
val_files = load_decathlon_datalist(datasets, True, "validation")
#train_files, val_files = partition_dataset(datalist, ratios=[0.8, 0.2], shuffle=False)

train_ds = CacheDataset(
    data=datalist,
    transform=train_transforms,
    cache_num=12,
    cache_rate=1.0,
    num_workers=8
)
train_loader = ThreadDataLoader(train_ds, num_workers=0, batch_size=1, shuffle=True)

val_ds = CacheDataset(
    data=val_files,
    transform=val_transforms,
    cache_num=4,
    cache_rate=1.0,
    num_workers=4
)
val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1)


set_track_meta(False)

    


###-------------------------------------------Modeling------------------------------------------

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = SwinUNETR(
    img_size=(320, 320, 64),
    in_channels=1,
    out_channels=2,
    feature_size=48
).to(device)

#weight = torch.load("D:\\severance\\model_swinvit.pt")
#model.load_from(weights=weight)
#print("Using pretrained self-supervied Swin UNETR backbone weights !")

torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.nn.functional as F


class FocalDiceLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, softmax=True):
        super(FocalDiceLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.softmax = softmax

    def forward(self, inputs, targets, smooth=1e-5):
        if self.softmax:
            inputs = torch.softmax(inputs, dim=1)
        if targets.dim() == 5 and targets.shape[1] == 1:
            targets = targets.squeeze(1)
        targets = torch.nn.functional.one_hot(targets.long(), num_classes=inputs.shape[1]).permute(0, 4, 1, 2, 3).float()

        # Focal Loss
        if self.softmax:
            BCE = F.cross_entropy(inputs, targets.argmax(dim=1), reduction='none')
        else:
            BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = self.alpha * (1 - BCE_EXP) ** self.gamma * BCE
        focal_loss = focal_loss.mean()

        # Dice Loss
        intersection = (inputs * targets).sum(dim=(2, 3, 4))
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum(dim=(2, 3, 4)) + targets.sum(dim=(2, 3, 4)) + smooth)
        dice_loss = dice_loss.mean()

        
        total_loss = focal_loss + dice_loss

        return total_loss

#lr scheduler
initial_lr = 1e-4
final_lr = 1e-6
max_epochs = 35000
lambda_lr = lambda epoch: (final_lr / initial_lr) ** (epoch / max_epochs)


#loss_function = FocalDiceLoss(gamma=3.5, alpha=0.85, softmax=True)
#loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
weights = torch.tensor([0.5, 5.0], dtype=torch.float).cuda()
loss_function = nn.CrossEntropyLoss(weight=weights)

optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-6)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)

scaler = torch.cuda.amp.GradScaler()


###------------------------------------Train/Validation define-------------------------------------------

def validation(epoch_iterator_val, last_fc_size, is_text=False):
    model.eval()
    torch.cuda.empty_cache()
    with torch.no_grad():
        for batch in epoch_iterator_val:
            if is_text:
                val_inputs, val_labels, text_path = (batch["image"].cuda(), batch["label"].cuda(), batch["text"])
                text_path = os.path.join(data_dir,text_path[0])
                val_text = torch.load(text_path).to(device)
                with torch.cuda.amp.autocast():
                    val_outputs = model(val_inputs, val_text)
            else:
                val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
                with torch.cuda.amp.autocast():
                    val_outputs = sliding_window_inference(val_inputs, (320, 320, 64), 1, model)

            if val_labels.max() >= last_fc_size:
                val_labels[val_labels >= last_fc_size] = 0
            
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            if val_output_convert is not None and val_labels_convert is not None:
                dice_metric(y_pred=val_output_convert, y=val_labels_convert)
                dice_metric_back(y_pred=val_output_convert, y=val_labels_convert)
                surface_distance_metric(y_pred=val_output_convert, y=val_labels_convert)
                iou_metric(y_pred=val_output_convert, y=val_labels_convert)
            else:
                print("Invalid data encountered in validation")
                continue
            epoch_iterator_val.set_description("Validate (%d / %d Steps)" % (global_step, 10.0))
        mean_dice_val = dice_metric.aggregate().item()
        mean_dice_val_back = dice_metric_back.aggregate().item()
        mean_asd_val = surface_distance_metric.aggregate().item()
        mean_iou_val = iou_metric.aggregate().item()
        dice_metric_back.reset()
        dice_metric.reset()
        surface_distance_metric.reset()
        iou_metric.reset()

    return mean_dice_val,  mean_dice_val_back, mean_asd_val, mean_iou_val


def train(global_step, train_loader, dice_val_best, global_step_best, last_fc_size, is_text=False):
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
    for step, batch in enumerate(epoch_iterator):
        step += 1
        
        if is_text:
            x, y, text_path = (batch["image"].cuda(), batch["label"].cuda(), batch["text"][0])
            y=y.squeeze(1).long() #WCELoss일때만
            text_path = os.path.join(data_dir,text_path)
            text = torch.load(text_path).to(device)

            with torch.cuda.amp.autocast():
                logit_map = model(x, text)
                if logit_map is not None and y is not None:
                    loss = loss_function(logit_map, y)
                else:
                    print("Invalid data encountered in training")
                    continue
        
        else:
            x,y = (batch["image"].cuda(), batch["label"].cuda())
            y=y.squeeze(1).long() #WCELoss일때만
            with torch.cuda.amp.autocast():
                logit_map = model(x)
                if logit_map is not None and y is not None:
                    loss = loss_function(logit_map, y)
                else:
                    print("Invalid data encountered in training")
                    continue
        
        if y.max() >= last_fc_size:
            print("\nReplace labels >= {} to 0".format(last_fc_size))
            y[y >= last_fc_size] = 0
        

        scaler.scale(loss).backward()


        total_norm=0
        for p in model.parameters():
            
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        epoch_loss += loss.item()
        
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()
        

        epoch_iterator.set_description(f"Training ({global_step} / {max_iterations} Steps) (loss={loss:2.5f})")
        if global_step == 30:
            epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
            dice_val,dice_back, asd_val, iou_val = validation(epoch_iterator_val, last_fc_size, is_text=True)
            print(
                    "Model Was Not Saved ! Current Avg. Dice: {} Dice Back : {} \n asd val: {} iou val:{}".format(dice_val,dice_back, asd_val, iou_val)
                )

        if (global_step % eval_num == 0 and global_step != 0) or global_step == max_iterations:
            epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
            dice_val,dice_back, asd_val, iou_val = validation(epoch_iterator_val, last_fc_size, is_text=True)
            epoch_loss /= step

            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)
            dice_backs.append(dice_back)
            asd_values.append(asd_val)
            iou_values.append(iou_val)

            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                torch.save(model.state_dict(), os.path.join(root_dir, f"{rand_num} best_metric_model.pth"))
                print(
                    "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {} Dice Back: {} \n  asd val: {} iou val:{}".format(dice_val_best, dice_val, dice_back, asd_val, iou_val)
                )
            else:
                print(
                    "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {} Dice Back: {} \n asd val: {} iou val:{}".format(dice_val_best, dice_val, dice_back, asd_val, iou_val)
                )
        global_step += 1
    return global_step, dice_val_best, global_step_best


###------------------------------------Training---------------------------------------
#내 y가 0,1,2를 넘는 경우를 pass하고 훈련진행하기위한 장치
last_fc_size = 3
max_iterations = 35000
eval_num = 500
post_label = AsDiscrete(to_onehot=2)
post_pred = AsDiscrete(argmax=True, to_onehot=2)
dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
dice_metric_back = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
#hausdorff_distance_metric = HausdorffDistanceMetric(include_background=True, percentile=80)
surface_distance_metric = SurfaceDistanceMetric(include_background=True)
iou_metric = MeanIoU(include_background=True)

global_step = 0
dice_val_best = 0.0
global_step_best = 0

#예제 llm feature

epoch_loss_values = []
metric_values = []
hd_values = []
asd_values = []
iou_values = []
dice_backs = []

'''
while global_step < max_iterations:
    global_step, dice_val_best, global_step_best = train(global_step, train_loader, dice_val_best, global_step_best, last_fc_size, is_text=True)
print('epoch_loss_values:',epoch_loss_values, 'metric_values:',metric_values, 'asd_values: ',asd_values, 'iou_Values: ',iou_values)
'''
import json

# Data to be saved
data = {
    "epoch_loss_values": epoch_loss_values,
    "metric_values": metric_values,
    "asd_values": asd_values,
    "iou_values": iou_values
}

# Convert to JSON string
json_data = json.dumps(data, indent=4)

# Write to file
#with open('D:\\Tumor_Segmentation\\LiTS\\models\\SwinUnetr\\metrics.json', 'w') as file:
#    file.write(json_data)



plt.figure("Performance", (18, 6))

# Plot for iteration average loss
plt.subplot(1, 3, 1)
plt.title("Iteration Average Loss")
x = [eval_num * (i + 1) for i in range(len(epoch_loss_values))]
plt.xlabel("Iteration")
plt.plot(x, epoch_loss_values, label='Loss')
plt.legend()

# Plot for validation mean dice
plt.subplot(1, 3, 2)
plt.title("Val Mean Dice")
x = [eval_num * (i + 1) for i in range(len(metric_values))]
plt.xlabel("Iteration")
plt.plot(x, metric_values, label='Dice')
plt.legend()

# Plot for additional metrics
plt.subplot(1, 3, 3)
plt.title("Validation Metrics")
x = [eval_num * (i + 1) for i in range(len(asd_values))]
plt.xlabel("Iteration")
plt.plot(x, asd_values, label='Average Surface Distance')
plt.plot(x, iou_values, label='IoU')
plt.legend()


plt.show()
case_num = [12, 5, 7]
model.load_state_dict(torch.load(os.path.join(root_dir, f"tumor-skip best_metric_model.pth")))
model.eval()
with torch.no_grad():
    for num in case_num:
        img = val_ds[num]["image"]
        label = val_ds[num]["label"]
        text = val_ds[num]["text"]
        print(text)
        #torch.round(label)
        val_inputs = torch.unsqueeze(img, 1).cuda().float()
        print(val_inputs.shape)
        val_labels = torch.unsqueeze(label, 1).cuda().float()
        text_path = os.path.join(data_dir, text)
        val_text = torch.load(text_path).to(device).float()
        val_outputs = model(val_inputs, val_text)
        #val_outputs = sliding_window_inference(val_inputs, (320, 128, 128), 4, model, overlap=0.8)
        plt.figure("check", (18, 6))
        plt.subplot(1, 3, 1)
        plt.title("image")
        plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, 30], cmap="gray")
        plt.subplot(1, 3, 2)
        plt.title("label")
        plt.imshow(val_labels.cpu().numpy()[0, 0, :, :, 30])
        plt.subplot(1, 3, 3)
        plt.title("output")
        plt.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, 30])
        plt.show()