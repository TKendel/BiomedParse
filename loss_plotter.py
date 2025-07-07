import os
import json
import regex as re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



# paths = ['slurm_output_FULL_CT.out','slurm_output_FULL_CT_2.out', 'slurm_output_FULL_CT_3.out','slurm_output_FULL_CT_4.out','slurm_output_FULL_CT_5.out','slurm_output_FULL_CT_6.out','slurm_output_FULL_CT_7.out']
paths = ['FINAL_OUTPUTS_SLURM\slurm-1374654.out']
# dir_name = 'scans-40_MRI_fr=5,r=8,a=16,lr=0.00005,drop_path=0.3,drop=0.1'
dir_name = 'CHECK'

anchor = "INFO:trainer.default_trainer:epochs["
anchor_2 = "INFO:datasets.evaluation.grounding_evaluation:"

loss_mask_ce = []
loss_mask_bce = []
loss_mask_dice = []
loss_spatial_bce = []
loss_spatial_dice = []
loss_spatial_ce = []
loss_grounding_bce = []
loss_grounding_dice = []
loss_grounding_ce = []

precision_5 = []
precision_6 = []
precision_7 = []
precision_8 = []
precision_9 = []
cIoU = []
mIoU = []
cDice = []
mDice = []

for path in paths:
    with open(path) as f:
        previous_line = None
        for line in f.readlines(): 
            # if anchor in line:

            mask_ce = []
            mask_bce = []
            mask_dice = []
            spatial_bce = []
            spatial_dice = []
            spatial_ce = []
            grounding_bce = []
            grounding_dice = []
            grounding_ce = []

            # Get epoch
            # if epoch == None:
            #     epoch = int(line[len(anchor):len(anchor)+6])
            #     continue
            # elif epoch == int(line[len(anchor):len(anchor)+6]):
            #     last_line = line
            #     continue
            # elif epoch != int(line[len(anchor):len(anchor)+6]):

            if anchor_2 in line:
                idx = line.index("{")
                validation = line[idx:]
                json_acceptable_string = validation.replace("'", "\"")
                validation = json.loads(json_acceptable_string)

                precision_5.append(validation['precision@0.5'])
                precision_6.append(validation['precision@0.6'])
                precision_7.append(validation['precision@0.7'])
                precision_8.append(validation['precision@0.8'])
                precision_9.append(validation['precision@0.9'])
                cIoU.append(validation['cIoU'])
                mIoU.append(validation['mIoU'])
                cDice.append(validation['cDice'])
                mDice.append(validation['mDice'])



            if 'INFO:trainer.default_trainer:Evaluation start ...' not in line and 'WARNING:trainer.utils_trainer:Saving checkpoint...' not in line:
                previous_line = line
            else:
                if previous_line != None:
                    if anchor in previous_line:
                        
                        # Get train loss list
                        idx_2 = previous_line.index('items per batch')
                        losses = previous_line[: idx_2]
                        # print(losses, "\n")
                        pattern = r'(loss_[a-z_0-9]+): ([0-9.]+)\/([0-9.]+)'

                        # Extract losses
                        results = re.findall(pattern, losses)

                        for result in results:
                            if 'mask_ce' in result[0]:
                                mask_ce.append(result[2])
                            elif 'mask_bce' in result[0]:
                                mask_bce.append(result[2])
                            elif 'mask_dice' in result[0]:
                                mask_dice.append(result[2])
                            elif 'spatial_ce' in result[0]:
                                spatial_ce.append(result[2])
                            elif 'spatial_bce' in result[0]:
                                spatial_bce.append(result[2])
                            elif 'spatial_dice' in result[0]:
                                spatial_dice.append(result[2])
                            elif 'grounding_ce' in result[0]:
                                grounding_ce.append(result[2])
                            elif 'grounding_bce' in result[0]:
                                grounding_bce.append(result[2])
                            elif 'grounding_dice' in result[0]:
                                grounding_dice.append(result[2])

                        loss_mask_ce.append(mask_ce)
                        loss_mask_bce.append(mask_bce)
                        loss_mask_dice.append(mask_dice)
                        loss_spatial_ce.append(spatial_ce)
                        loss_spatial_bce.append(spatial_bce)
                        loss_spatial_dice.append(spatial_dice)
                        loss_grounding_ce.append(grounding_ce)
                        loss_grounding_bce.append(grounding_bce)
                        loss_grounding_dice.append(grounding_dice)

                        previous_line = None

loss_list = {"loss_mask_ce": loss_mask_ce, "loss_mask_bce": loss_mask_bce, "loss_mask_dice": loss_mask_dice, "loss_spatial_ce": loss_spatial_ce, "loss_spatial_bce": loss_spatial_bce, "loss_spatial_dice": loss_spatial_dice, "loss_grounding_ce": loss_grounding_ce, "loss_grounding_bce": loss_grounding_bce, "loss_grounding_dice": loss_grounding_dice}
eval_list = {"precision_5": precision_5, "precision_6": precision_6, "precision_7": precision_7, "precision_8": precision_8, "precision_9": precision_9, "cIoU": cIoU, "mIoU": mIoU, "cDice": cDice, "mDice": mDice, "Epochs": len(precision_5)}


for loss_list_key in loss_list:
    print(loss_list_key)
    for layer in range(0, 10):
        loss_values = []
        for loss_name in loss_list[loss_list_key]:
            loss_values.append(float(loss_name[layer]))
        
        # Add in a title and axes labels
        plt.title(f'{loss_list_key}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        # Set the tick locations
        # plt.xticks(arange(0, 21, 2))

        # Plot and label the training and validation loss values
        plt.plot(loss_values)
        
        # # Display the plot
        # plt.legend(loc='best')

        if not os.path.isdir(f'loss_plots/{dir_name}'):
            os.mkdir(f'loss_plots/{dir_name}')
        if not os.path.isdir(f'loss_plots/{dir_name}/layer_{layer}'):
            os.mkdir(f'loss_plots/{dir_name}/layer_{layer}')
        if not os.path.isdir(f'loss_plots/{dir_name}/layer_{layer}/{loss_list_key}'):
            os.mkdir(f'loss_plots/{dir_name}/layer_{layer}/{loss_list_key}')

        plt.savefig(f'loss_plots/{dir_name}/layer_{layer}/{loss_list_key}/{loss_list_key}{layer}.png')

        plt.clf()


for eval_list_key in eval_list:
    print(eval_list_key)

    # Add in a title and axes labels
    plt.title(f'{eval_list_key}')
    plt.xlabel('Epochs')
    plt.ylabel('Score')

    # Plot and label the training and validation loss values
    plt.plot(eval_list[eval_list_key])

    plt.savefig(f'loss_plots/{dir_name}/{eval_list_key}.png')

    plt.clf()





# Add in a title and axes labels
plt.title('Thresholding Precison Validation on 400 MRI Scans')
plt.xlabel('Epochs')
plt.ylabel('Score')

# Plot and label the training and validation loss values
# plt.plot(eval_list['precision_5'], eval_list['precision_6'], eval_list['precision_7'], eval_list['precision_8'], eval_list['precision_9'])
plt.plot(eval_list['precision_5'], label='0.5')
plt.plot(eval_list['precision_6'], label='0.6')
plt.plot(eval_list['precision_7'], label='0.7')
plt.plot(eval_list['precision_8'], label='0.8')
plt.plot(eval_list['precision_9'], label='0.9')
plt.legend()
plt.xticks(range(0, 19+1))

plt.savefig(f'loss_plots/{dir_name}/combined_precission.png')
