import os
import re
import numpy as np
import random



# pattern = r"^([^_]*)"

# DIR1 = 'biomedparse_datasets\\CT_pancreatic_cancer\\train'
# patient_numbers_img = [name[:-15] for name in os.listdir(DIR1) if os.path.isfile(os.path.join(DIR1, name))]
# patient_numbers_img = np.array(patient_numbers_img)

# print(patient_numbers_img)

# DIR2 = 'biomedparse_datasets\\CT_pancreatic_cancer\\train_mask'
# patient_numbers_mask = [re.match(pattern, name).group(1) for name in os.listdir(DIR2) if os.path.isfile(os.path.join(DIR2, name))]
# patient_numbers_mask = np.array(patient_numbers_mask)

# print(patient_numbers_mask)

DIR1 = 'biomedparse_datasets\\CT_pancreatic_cancer\\train'
patient_numbers_img = [int(name[:6]) for name in os.listdir(DIR1) if os.path.isfile(os.path.join(DIR1, name))]
patient_numbers_img = np.array(patient_numbers_img)

# print(len(patient_numbers_img))

DIR2 = 'biomedparse_datasets\\CT_pancreatic_cancer\\test'
patient_numbers_mask = [int(name[:6]) for name in os.listdir(DIR2) if os.path.isfile(os.path.join(DIR2, name))]
patient_numbers_mask = np.array(patient_numbers_mask)

# print(len(patient_numbers_mask))

patients = np.unique(np.concatenate([patient_numbers_mask, patient_numbers_img]))

DIR3 = 'biomedparse_datasets\\CT_pancreatic_cancer\\test_mask'
DIR4 = 'biomedparse_datasets\\CT_pancreatic_cancer\\train_mask'

unique_pat = set()
patient_img = [name for name in os.listdir(DIR3) if os.path.isfile(os.path.join(DIR3, name))]
patient_img2 = [name for name in os.listdir(DIR4) if os.path.isfile(os.path.join(DIR4, name))]
patient_img = np.array(patient_img)
patient_img2 = np.array(patient_img2)

patient_paths = np.concatenate([patient_img2, patient_img])


for pth in patient_paths:
    if "tumor" in pth:
        unique_pat.add(int(pth[:6]))

# 80 20, manual holdout of first 500 

split_80_PDAC = int(len(unique_pat) * 0.8)
split_80 = int(len(patients) * 0.8)

print(len(patients))
print(len(unique_pat))

patients = [x for x in patients if x not in unique_pat]

print(patients)

train_label = np.random.choice(np.array(list(unique_pat)), size=split_80_PDAC)
train_raw = np.random.choice(np.array(list(patients)), size=split_80)



for r in train_label:
    for pth in patient_paths:
        if str(r) in pth:
            os.rename(f"{DIR1}\\{pth}")


# for patient in patient_numbers_img:
#     if patient not in patient_numbers_mask:
#         os.rename(f"{DIR1}\\{patient}_CT_abdomen.png", f"biomedparse_datasets\\CT_pancreatic_cancer\\trash_train\\{patient}_CT_abdomen.png")
    

# print(np.unique(patient_numbers_mask).shape, np.unique(patient_numbers_img).shape)

# print((np.unique(patient_numbers_mask)==np.unique(patient_numbers_img)).all())


# for p in patient_numbers_img:
#     if p not in patient_numbers_mask:
#         os.rename(f'biomedparse_datasets_CT\\CT_pancreatic_cancer\\train\\{p}_CT_abdomen.png', f'biomedparse_datasets_CT\\CT_pancreatic_cancer\\trash\\{p}_CT_abdomen.png')

# patient_img = [name for name in os.listdir(DIR2) if os.path.isfile(os.path.join(DIR2, name))]
# patient_img = np.array(patient_img)
# print(patient_img)
# for pth in patient_img:
#     if "pancreas+tumor" in pth:
#         new_name = pth.replace("abdomen_pancreas+tumor", 'pancreas_tumor')
#         os.rename(f"biomedparse_datasets\\CT_pancreatic_cancer\\train_mask\\{pth}", f"biomedparse_datasets\\CT_pancreatic_cancer\\train_mask\\{new_name}") 
# pancreatic+arteries, pancretic+veins,pacreatic+parenchyma


# for f in os.listdir(DIR):
#     if os.path.isfile(os.path.join(DIR, f)):
#         _count = [m.start() for m in re.finditer('_', f)]
#         f_list = list(f)
#         f_list[_count[0]] = "-"

#         new_name = "".join(f_list)
#         print(f)
#         os.rename(f"biomedparse_datasets\\CT_pancreatic_cancer\\test\\{f}", f"biomedparse_datasets\\CT_pancreatic_cancer\\test\\{new_name}") 

