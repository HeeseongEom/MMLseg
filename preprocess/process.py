import json
import re
import os
import torch
from torch.nn.functional import normalize

"""def normalize_text_embeddings(src_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir, exist_ok=True)

    texts = sorted(os.listdir(src_dir), key=lambda x: int(x.split(".")[0]))
    for text_file in texts:
        text_path = os.path.join(src_dir, text_file)
        text_embedding = torch.load(text_path)

        normalized_embedding = normalize(text_embedding, p=2, dim=-1)

        save_path = os.path.join(dst_dir, text_file)
        torch.save(normalized_embedding, save_path)

pt_path = "D:\\MultiModal\\research\\research1\\data\\text"

new_pt_path = "D:\\MultiModal\\research\\research1\\data\\normalized_text"

"""


import nibabel as nib
import numpy as np
import os

def crop_or_pad_to_target_whd(img_array, label_array, target_w=320, target_h=320, margin=3):
   
    height, width, depth = img_array.shape
    liver_indices = np.argwhere(np.round(label_array) == 1)
    #print(np.unique(label_array))
    if liver_indices.size == 0:
        print("Liver not found. Using original image dimensions.")
        return img_array
    
    min_h, min_w, min_d = np.min(liver_indices, axis=0)
    max_h, max_w, max_d = np.max(liver_indices, axis=0)
    
    start_d = max(min_d - margin, 0)
    end_d = min(max_d + margin, depth)
    
    cropped_img = np.zeros((target_h, target_w, depth), dtype=img_array.dtype)
    cropped_lab = np.zeros((target_h, target_w, depth), dtype=label_array.dtype)
    
    center_w = width // 2
    center_h = height // 2
    start_w = max(center_w - target_w // 2, 0)
    start_h = max(center_h - target_h // 2, 0)
    
    end_w = start_w + target_w
    end_h = start_h + target_h
    
    cropped_img[:min(target_h, height), :min(target_w, width), :] = img_array[start_h:end_h, start_w:end_w, :]
    cropped_lab[:min(target_h, height), :min(target_w, width), :] = label_array[start_h:end_h, start_w:end_w, :]
    return cropped_img, cropped_lab

def process_images(src_dir, dst_dir, target_w=320, target_h=320, margin=3):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir, exist_ok=True)

    for patient_folder in sorted(os.listdir(src_dir)):
        if (patient_folder>'62') & (patient_folder!='87'):
            src_patient_path = os.path.join(src_dir, patient_folder)
            dst_patient_path = os.path.join(dst_dir, patient_folder)
            
            if not os.path.exists(dst_patient_path):
                os.makedirs(dst_patient_path, exist_ok=True)
                
            label_path = os.path.join(src_patient_path, "label.nii.gz")
            data_path = os.path.join(src_patient_path, "data.nii.gz")

            data_img = nib.load(data_path)
            label_img = nib.load(label_path)

            data_array = data_img.get_fdata()
            label_array = label_img.get_fdata()

            cropped_data, cropped_label = crop_or_pad_to_target_whd(data_array, label_array, target_w, target_h, margin)

            nib.save(nib.Nifti1Image(cropped_data, affine=data_img.affine), os.path.join(dst_patient_path, "data.nii.gz"))
            nib.save(nib.Nifti1Image(cropped_label, affine=label_img.affine), os.path.join(dst_patient_path, "label.nii.gz"))



ori_dir = "D:\\MultiModal\\research\\research1\\data\\nii_data_ori"
dst_dir = "D:\\MultiModal\\research\\research1\\data\\nii_data_crop"

process_images(ori_dir, dst_dir)

import os

'''def renumber_folders(base_dir):
    folders = sorted([f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))], key=int)
    
    temp_name = "temp_{}"
    for i, folder in enumerate(folders):
        original_path = os.path.join(base_dir, folder)
        temp_path = os.path.join(base_dir, temp_name.format(i))
        os.rename(original_path, temp_path)
    
    for i in range(len(folders)):
        temp_path = os.path.join(base_dir, temp_name.format(i))
        final_path = os.path.join(base_dir, str(i))
        os.rename(temp_path, final_path)
        
    print(f"Renumbered {len(folders)} folders.")'''

base_dir = "D:\\MultiModal\\research\\research1\\data\\nii_data"
ratios = []


import nibabel as nib
import numpy as np
patients = sorted(os.listdir(base_dir), key= lambda x : int(x))
list= []
for patient in patients:
    patient_dir = os.path.join(base_dir, patient)
    files = os.listdir(patient_dir)
    label_dir = [f for f in files if f.startswith('label')]

    label = nib.load(os.path.join(patient_dir, label_dir[0]))
    label_data = np.round(label.get_fdata())
    num_zeros = np.count_nonzero(label_data == 0)
    num_ones = np.count_nonzero(label_data == 1)
    
    if num_ones == 0:  
        continue
    
    ratio = num_zeros / num_ones
    ratios.append(ratio)

average_ratio = np.mean(ratios)
print(average_ratio)

'''import json
import re

def renumber_json_dataset(json_path):

    with open(json_path, 'r') as file:
        data = json.load(file)
    
    for key in ['training', 'validation']:
        new_list = []
        for i, item in enumerate(data[key]):
            for field in ['image', 'label', 'text']:
                old_num = int(re.search(r'(\d+)', item[field]).group(0))
                new_num = i  
                item[field] = item[field].replace(str(old_num), str(new_num))
            new_list.append(item)
        
        data[key] = new_list
    
    new_json_path = json_path.replace('dataset.json', 'dataset_renumbered.json')
    with open(new_json_path, 'w') as file:
        json.dump(data, file, indent=4)
    
    print(f"Renumbered dataset saved to {new_json_path}")

json_path = "D:\\MultiModal\\research\\research1\\data\\dataset.json"
renumber_json_dataset(json_path)'''