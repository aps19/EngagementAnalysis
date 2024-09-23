# utils/visualization.py

import matplotlib.pyplot as plt

def plot_metrics(history, metrics=['loss', 'accuracy'], save_path=None):
    """
    Plots training and validation metrics over epochs.

    Args:
        history (dict): Dictionary containing 'train' and 'val' metrics.
        metrics (List[str]): List of metric names to plot.
        save_path (str, optional): Path to save the plot image. If None, shows the plot.
    """
    epochs = range(1, len(history['train'][metrics[0]]) + 1)
    plt.figure(figsize=(12, 5))

    for idx, metric in enumerate(metrics):
        plt.subplot(1, len(metrics), idx + 1)
        plt.plot(epochs, history['train'][metric], 'b-', label=f'Train {metric.capitalize()}')
        plt.plot(epochs, history['val'][metric], 'r-', label=f'Val {metric.capitalize()}')
        plt.title(f'Training and Validation {metric.capitalize()}')
        plt.xlabel('Epochs')
        plt.ylabel(metric.capitalize())
        plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
