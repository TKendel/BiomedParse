import os
import re
import numpy as np

pattern = r"^([^_]*)"

DIR1 = 'biomedparse_datasets\\CT_pancreatic_cancer\\train'
patient_numbers_img = [name[:-15] for name in os.listdir(DIR1) if os.path.isfile(os.path.join(DIR1, name))]
patient_numbers_img = np.array(patient_numbers_img)

print(patient_numbers_img)

DIR2 = 'biomedparse_datasets\\CT_pancreatic_cancer\\train_mask'
patient_numbers_mask = [re.match(pattern, name).group(1) for name in os.listdir(DIR2) if os.path.isfile(os.path.join(DIR2, name))]
patient_numbers_mask = np.array(patient_numbers_mask)

print(patient_numbers_mask)

# for patient in patient_numbers_img:
#     if patient not in patient_numbers_mask:
#         os.rename(f"{DIR1}\\{patient}_CT_abdomen.png", f"biomedparse_datasets\\CT_pancreatic_cancer\\trash_train\\{patient}_CT_abdomen.png")
    

print(np.unique(patient_numbers_mask).shape, np.unique(patient_numbers_img).shape)

print((np.unique(patient_numbers_mask)==np.unique(patient_numbers_img)).all())


# for p in patient_numbers_img:
#     if p not in patient_numbers_mask:
#         os.rename(f'biomedparse_datasets_CT\\CT_pancreatic_cancer\\train\\{p}_CT_abdomen.png', f'biomedparse_datasets_CT\\CT_pancreatic_cancer\\trash\\{p}_CT_abdomen.png')

# patient_img = [name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]
# patient_img = np.array(patient_img)
# pancreatic+arteries, pancretic+veins,pacreatic+parenchyma


# for f in os.listdir(DIR):
#     if os.path.isfile(os.path.join(DIR, f)):
#         _count = [m.start() for m in re.finditer('_', f)]
#         f_list = list(f)
#         f_list[_count[0]] = "-"

#         new_name = "".join(f_list)
#         print(f)
#         os.rename(f"biomedparse_datasets\\CT_pancreatic_cancer\\test\\{f}", f"biomedparse_datasets\\CT_pancreatic_cancer\\test\\{new_name}") 

