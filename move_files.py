import os
import re
import numpy as np
import random



DIR1 = 'biomedparse_datasets\\AUMC\\train'
patient_numbers_img = [int(name[:3]) for name in os.listdir(DIR1) if os.path.isfile(os.path.join(DIR1, name))]
patient_numbers_img = np.array(patient_numbers_img)

# print(len(patient_numbers_img))

DIR2 = 'biomedparse_datasets\\AUMC\\train_mask'
patient_numbers_mask = [int(name[:3]) for name in os.listdir(DIR2) if os.path.isfile(os.path.join(DIR2, name))]
patient_numbers_mask = np.array(patient_numbers_mask)

# print(len(patient_numbers_mask))

# Get path of files
patient_img_label = [name for name in os.listdir(DIR2) if os.path.isfile(os.path.join(DIR2, name))]
patient_img_label = np.array(patient_img_label)
patient_img = [name for name in os.listdir(DIR1) if os.path.isfile(os.path.join(DIR1, name))]
patient_img = np.array(patient_img)

# Remove duplicates
patients = np.unique(patient_numbers_img)

# Split number
split_90 = int(len(patients) * 0.9)
split_10 = len(patients) - split_90

print(len(patients))
print(split_10, split_90)

# Split
train = np.random.choice(np.array(list(patients)), size=split_90, replace=False)

test = []
for p in patients:
    if p not in train:
        test.append(p)

print("----------------")
print(len(test), len(train))
print(len(np.unique(test)), len(np.unique(train)))

# Check for possible duplicates
if not np.intersect1d(train, test):
    # Sample
    for num in test:
        for pth in patient_img_label:
            if str(num) in pth:
                os.rename(f"{DIR2}\\{pth}", f"biomedparse_datasets\\AUMC\\test_mask\\{pth}")
        for pth in patient_img:
            if str(num) in pth:
                os.rename(f"{DIR1}\\{pth}", f"biomedparse_datasets\\AUMC\\test\\{pth}")
