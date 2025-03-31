import numpy as np
import re


def avg_metrics(lines):
    metrics_array = []

    for line in lines:
        metrics_array.append(re.search("(?<=:).*", line).group(0))

    metrics_array = np.array(metrics_array).astype(float)

    return np.average(metrics_array)



with open("metrics_baseline.txt", "r") as f:
    metrics_lines = f.readlines()

DICE = metrics_lines[0::4]
IoU = metrics_lines[1::4]
HD = metrics_lines[2::4]

DICE_avg = avg_metrics(DICE)
IoU_avg = avg_metrics(IoU)
HD_avg = avg_metrics(HD)

print(f"DICE average over {len(DICE)} patiens is: {DICE_avg}")
print(f"IoU average over {len(IoU)} patiens is: {IoU_avg}")
print(f"HD average over {len(HD)} patiens is: {HD_avg}")
