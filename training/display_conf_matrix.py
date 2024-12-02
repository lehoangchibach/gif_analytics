import json
import sys

import matplotlib.pyplot as plt
import seaborn as sns

try:
    conf_matrix_path = sys.argv[1]
except:
    raise Exception("Need a confusion matrix path.")

conf_matrix_name = conf_matrix_path.split("/")[-1].split(".")[0]

with open(conf_matrix_path, "r") as file:
    confusion_matrix = json.load(file)

# Create a heatmap using seaborn
fig_size = (40, 40)
plt.figure(figsize=fig_size)
sns.heatmap(confusion_matrix, annot=False, fmt="d", cmap="Blues", cbar=True)

# Add labels and title
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")

# Show the plot
plt.savefig(f"./outputs/{conf_matrix_name}.png")
plt.savefig(f"./outputs/{conf_matrix_name}.svg", format="svg")
