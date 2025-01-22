import json
import sys

import matplotlib.pyplot as plt


def plot_network_history(network_history, start=0, end=None):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    axes[0].plot(network_history["train_losses"][start:end], label="Training Loss")
    axes[0].plot(network_history["val_losses"][start:end], label="Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(
        network_history["train_accs"][start:end],
        label="Training Categorical Accuracy",
    )
    axes[1].plot(
        network_history["val_accs"][start:end],
        label="Validation Categorical Accuracy",
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    # plt.show()
    output = "./outputs/training_history.png"
    plt.savefig("./outputs/training_history.png")
    print(f"Save history image to {output}")


path = sys.argv[1]
with open(path) as f:
    stats = json.load(f)
plot_network_history(stats, start=0, end=None)
