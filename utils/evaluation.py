import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix', save_path=None, normalize=False):
    """
    Generates and plots a Confusion Matrix heatmap.

    Args:
        y_true (list): Ground truth labels.
        y_pred (list): Predicted labels.
        classes (list): List of class names (e.g., ['ABB', 'Kapital', ...]).
        title (str): Title of the plot.
        save_path (str): File path to save the image (e.g., 'reports/cm.png').
        normalize (bool): If True, shows percentages instead of raw counts.
    """
    # Compute matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = f"{title} (Normalized)"
    else:
        fmt = 'd'

    # Setup Plot
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.2)
    
    # Draw Heatmap
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    
    plt.title(title, fontweight='bold', pad=20)
    plt.ylabel('Actual Class', fontweight='bold')
    plt.xlabel('Predicted Class', fontweight='bold')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save or Show
    if save_path:
        # Ensure folder exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close() # Close to free memory
        print(f"   Saved Plot: {save_path}")
    else:
        plt.show()

def plot_training_loss(logs, save_path=None):
    """
    Optional helper to plot loss curves if you have the data logs.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(logs['train_loss'], label='Train Loss')
    plt.plot(logs['val_loss'], label='Val Loss')
    plt.title("Training Performance")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()