import os
import re
import numpy as np
import random
import nibabel as nib



def check_diff(DIR1, DIR2):
    pattern = r"^([^_]*)"

    DIR1 = 'biomedparse_datasets\\CT_pancreas\\train'
    patient_numbers_img = [name[:-15] for name in os.listdir(DIR1) if os.path.isfile(os.path.join(DIR1, name))]
    patient_numbers_img = np.array(patient_numbers_img)

    print(patient_numbers_img)

    DIR2 = 'biomedparse_datasets\\CT_pancreas\\train_mask'
    patient_numbers_mask = [re.match(pattern, name).group(1) for name in os.listdir(DIR2) if os.path.isfile(os.path.join(DIR2, name))]
    patient_numbers_mask = np.array(patient_numbers_mask)

    print(patient_numbers_mask)

    print(np.unique(patient_numbers_mask).shape, np.unique(patient_numbers_img).shape)
    print((np.unique(patient_numbers_mask)==np.unique(patient_numbers_img)).all())



def rename_files(DIR, old, new):
    patient_img = [name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]
    patient_img = np.array(patient_img)
    print(patient_img)
    for pth in patient_img:
        if old in pth:
            new_name = pth.replace(old, new)
            os.rename(f"biomedparse_datasets\\CT_pancreatic_cancer\\train_mask\\{pth}", f"biomedparse_datasets\\CT_pancreatic_cancer\\train_mask\\{new_name}") 


def random_sample(DIR1, DIR2):
    random.seed(90)

    patient_img_label = [name for name in os.listdir(DIR2) if os.path.isfile(os.path.join(DIR2, name))]
    patient_img_label = np.array(patient_img_label)
    patient_img = [name for name in os.listdir(DIR1) if os.path.isfile(os.path.join(DIR1, name))]
    patient_img = np.array(patient_img)

    unique_pat = set()

    for pth in patient_img_label:
        if "tumor" in pth:
            unique_pat.add(int(pth[:6]))

    unique_pat = list(unique_pat)

    PDAC_patients = [100002, 100005, 100011, 100030, 100033, 100043, 100050, 100060, 100074, 100082, 100091, 100096, 100101, 100102, 100124, 100127, 100134]

    print(len(patients))

    patients = [x for x in patients if x not in unique_pat]

    # Remove specific patients for personal testing from unique list of tumor patients
    for PDAC in PDAC_patients:
        index = np.argwhere(unique_pat==PDAC)
        unique_pat = np.delete(unique_pat, index)

    # Get 70 split for both
    split_70_PDAC = int(len(unique_pat) * 0.7)
    split_70 = int(len(patients) * 0.7)

    print(len(patients))
    print(len(unique_pat))

    print(split_70, split_70_PDAC)

    # Get test split from remaining
    train_healthy = np.random.choice(np.array(list(patients)), size=split_70, replace=False)
    train_PDAC = np.random.choice(np.array(list(unique_pat)), size=split_70_PDAC, replace=False)

    test_healthy = []
    test_PDAC = []

    for p in patients:
        if p not in train_healthy:
            test_healthy.append(p)
    for p in unique_pat:
        if p not in train_PDAC:
            test_PDAC.append(p)


    print("----------------")
    print(len(train_healthy), len(train_PDAC))
    print(len(np.unique(train_healthy)), len(np.unique(train_PDAC)))
    print(len(test_healthy), len(test_PDAC))
    print(len(np.unique(test_healthy)), len(np.unique(test_PDAC)))

    # Check for possible duplicates
    print(np.intersect1d(train_healthy, train_PDAC))
    print(np.intersect1d(train_healthy,test_healthy))
    print(np.intersect1d(train_healthy,test_PDAC))
    print(np.intersect1d(test_healthy,test_PDAC))
    print(np.intersect1d(test_healthy,train_PDAC))
    print(np.intersect1d(train_PDAC, test_PDAC))

    # Fill train_masks with patients from train_lable and train_healthy
    for num in train_PDAC:
        for pth in patient_img_label:
            if str(num) in pth:
                os.rename(f"{DIR2}\\{pth}", f"biomedparse_datasets\\CT_pancreas\\train_mask\\{pth}")
        for pth in patient_img:
            if str(num) in pth:
                os.rename(f"{DIR1}\\{pth}", f"biomedparse_datasets\\CT_pancreas\\train\\{pth}")

    for num in train_healthy:
        for pth in patient_img:
            if str(num) in pth:
                os.rename(f"{DIR1}\\{pth}", f"biomedparse_datasets\\CT_pancreas\\train\\{pth}")
        for pth in patient_img_label:
            if str(num) in pth:
                os.rename(f"{DIR2}\\{pth}", f"biomedparse_datasets\\CT_pancreas\\train_mask\\{pth}")

    for num in test_PDAC:
        for pth in patient_img_label:
            if str(num) in pth:
                os.rename(f"{DIR2}\\{pth}", f"biomedparse_datasets\\CT_pancreas\\test_mask\\{pth}")
        for pth in patient_img:
            if str(num) in pth:
                os.rename(f"{DIR1}\\{pth}", f"biomedparse_datasets\\CT_pancreas\\test\\{pth}")
                
    for num in test_healthy:
        for pth in patient_img:
            if str(num) in pth:
                os.rename(f"{DIR1}\\{pth}", f"biomedparse_datasets\\CT_pancreas\\test\\{pth}")
        for pth in patient_img_label:
            if str(num) in pth:
                os.rename(f"{DIR2}\\{pth}", f"biomedparse_datasets\\CT_pancreas\\test_mask\\{pth}")


def merge_labels(DIR):
    directories = os.listdir(DIR)

    PDAC_patients = ['010-5', '022-4', '036-4', '043-5', '062-5', '066-3', '073-5']

    label_dict = {"GTV": 1, "BOWEL": 2, "DUODENUM": 3, "LIVER": 4, "STOMACH": 5, "SPLEEN": 6, "KIDNEY_R": 7, "KIDNEY_L": 8}


    g = 0
    b=0
    d=0
    l=0
    s=0
    spl = 0
    kl = 0
    kr = 0
    for directory in directories:
        # if directory in PDAC_patients:
            sub_directories = os.listdir(f"{DIR}{directory}")
            for sub_directory in sub_directories:
                if sub_directory not in PDAC_patients:
                    files = os.listdir(f"{DIR}{directory}\\{sub_directory}")
                    gtv = None
                    bowel = None
                    duodenum = None
                    liver = None
                    stomach = None
                    for file in files:
                        print(f'{DIR}{directory}\\{sub_directory}\\{file}')
                        if 'mask_GTV' in file:
                            gtv = f'{DIR}{directory}\\{sub_directory}\\{file}'
                            g+=1
                        elif 'mask_BOWEL.nii' in file:
                            bowel = f'{DIR}{directory}\\{sub_directory}\\{file}'
                            b+=1
                        elif 'mask_DUODENUM.nii' in file:
                            duodenum = f'{DIR}{directory}\\{sub_directory}\\{file}'
                            d+=1
                        elif 'mask_LIVER.nii' in file:
                            liver = f'{DIR}{directory}\\{sub_directory}\\{file}'
                            l+=1
                        elif 'mask_STOMACH.nii' in file:
                            stomach = f'{DIR}{directory}\\{sub_directory}\\{file}'
                            s+=1
                        # elif 'mask_SPLEEN.nii' in file:
                        #     spleen = f'{DIR}{directory}\\{sub_directory}\\{file}'
                        #     spl+=1
                        # elif 'mask_KIDNEY_L.nii' in file or 'mask_KIDNEY_CONTRALAT' in file: 
                        #     kidney_l = f'{DIR}{directory}\\{sub_directory}\\{file}'
                        #     kl+=1
                        # elif 'mask_KIDNEY_R.nii' in file or 'mask_KIDNEY_IPSILATER' in file: 
                        #     kidney_r = f'{DIR}{directory}\\{sub_directory}\\{file}'
                        #     kr+=1

                    # if gtv and bowel and duodenum and liver and stomach:
                    #     pass
                    # else:
                    #     print(sub_directory)

                    gtv_affine = nib.load(gtv).affine
                    gtv = nib.load(gtv).get_fdata()
                    bowel = nib.load(bowel).get_fdata()
                    duodenum = nib.load(duodenum).get_fdata()
                    liver = nib.load(liver).get_fdata()
                    stomach = nib.load(stomach).get_fdata()

                    label_map = np.zeros_like(gtv)

                    label_map[gtv > 0] = 1
                    label_map[bowel > 0] = 2
                    label_map[duodenum > 0] = 3
                    label_map[liver > 0] = 4
                    label_map[stomach > 0] = 5

                    merged = nib.Nifti1Image(label_map.astype(np.uint8), gtv_affine)
                    nib.save(merged, f'{DIR}{directory}/{sub_directory}/mask_MERGED.nii.gz')

                    gtv = None
                    bowel = None
                    duodenum = None
                    liver = None
                    stomach = None

merge_labels("data/AUMC/")