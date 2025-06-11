# ResNet-Based Waste Image Classifier & Neural Collapse Analysis

This project focuses on building an image classifier to detect waste categories using a ResNet18 model and analyzing learned representations via t-SNE for signs of Neural Collapse.

## Project Objectives

- Train a deep learning model (ResNet18) on the TrashNet dataset (or CIFAR-10 for experimentation).
- Enhance model robustness with data augmentation techniques.
- Visualize feature space to interpret model behavior.
- Detect early-stage signs of Neural Collapse during training.

## Key Features

- Transfer learning with ResNet18 for high classification accuracy.
- Data augmentation (rotation, flipping, brightness adjustment) to avoid overfitting.
- Feature space visualization using t-SNE.
- Interpretation of intra-class collapse and its relevance to model generalization.

## Tech Stack

- Python
- PyTorch
- Torchvision
- NumPy, pandas
- matplotlib, seaborn
- scikit-learn (for t-SNE)

## Results Summary

| Metric       | Value        |
|--------------|--------------|
| Accuracy     | 73.68%       |
| Observations | Neural Collapse (Partial) detected via t-SNE plots |


## Sample t-SNE Plot

*Visualizes how well-separated the class features are in the final layers of ResNet.*

## Future Work

- Extend to real-world TrashNet images.
- Compare feature collapse across different architectures.
- Explore practical use in recycling stations.


