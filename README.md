# Exploring TensorFlow Callbacks in CNN Training

As a deep learning enthusiast, I developed this project to deepen my understanding of TensorFlow/Keras callbacks by applying them to a simple image classification task. This notebook demonstrates practical implementations of various callbacks, showcasing their mechanics and effects on model training. It's designed to highlight my skills in model building, callback customization, and training optimization, even if some runs yield moderate accuracies—intentionally kept to illustrate learning curves and callback behaviors for benchmarking purposes.

## Problem Statement and Goal of Project

The core task is binary image classification: distinguishing between horses and humans using a convolutional neural network (CNN). However, the primary goal is to explore and implement TensorFlow/Keras callbacks to monitor, control, and optimize the training process. This includes handling overfitting, adjusting learning rates, saving checkpoints, and creating custom callbacks for deeper insights into training dynamics.

## Solution Approach

- **Data Preparation**: Load and preprocess the 'horses_or_humans' dataset, resizing images to 150x150 and normalizing pixel values.
- **Model Architecture**: Build a sequential CNN with three convolutional layers (16, 32, 64 filters), max pooling, flattening, a dense layer (256 units), and a softmax output for 2 classes.
- **Training with Callbacks**:
  - TensorBoard for logging and visualization.
  - ModelCheckpoint for saving models at epochs.
  - EarlyStopping to halt training if validation loss doesn't improve.
  - LearningRateScheduler for step decay and exponential decay.
  - Custom callbacks for overfitting detection and detailed logging of batch/epoch events.
- Compile with SGD optimizer and sparse categorical crossentropy loss.
- Train on batched datasets, evaluating callback impacts through multiple runs.

## Technologies & Libraries

- TensorFlow (version 2.19.0) and Keras for model building and training.
- tensorflow_datasets for loading the dataset.
- Matplotlib and NumPy for data handling and visualization.
- Other: os, shutil, datetime, pandas, math.

## Description about Dataset

The 'horses_or_humans' dataset from tensorflow_datasets is used, containing 1027 training examples split into train (80%), validation (20%), and test sets. It has 2 classes (horses and humans), with RGB images originally sized variably but resized to 150x150x3 during preprocessing.

## Installation & Execution Guide

1. Ensure Python 3.12+ is installed with Conda or virtualenv.
2. Install dependencies:
   ```
   pip install tensorflow tensorflow-datasets matplotlib numpy pandas
   ```
3. Download and run the notebook:
   ```
   jupyter notebook callbacks_me.ipynb
   ```
4. For TensorBoard visualization:
   ```
   tensorboard --logdir logs --port 6006
   ```
   Access at http://localhost:6006.

## Key Results / Performance

- Model achieves accuracies up to ~0.98 on validation in some runs with callbacks applied.
- EarlyStopping example: Training stops at epoch 15 with val_loss ~0.095, restoring best weights from epoch 12.
- LearningRateScheduler (step decay): Learning rate halves every epoch, reducing val_loss progressively.
- Custom callback logs show training halts on overfitting detection (e.g., val/train loss ratio >0.7).
- TensorBoard logs demonstrate metrics like accuracy improving from ~0.50 to ~0.96 over 10 epochs.

These results emphasize callback utility in preventing overfitting and optimizing training, even with moderate initial accuracies to showcase iterative improvements.

## Screenshots / Sample Outputs

Sample training output with EarlyStopping:
```
Epoch 1/50
26/26 - 1s - 47ms/step - accuracy: 0.6058 - loss: 0.6705 - val_accuracy: 0.5756 - val_loss: 0.6592
...
Epoch 15: early stopping
Restoring model weights from the end of the best epoch: 12.
```

Custom callback logging example:
```
...Training: start of batch 0; got log keys: []
...Training: end of batch 0; got log keys: ['accuracy', 'loss']
...
End epoch 0 of training; got log keys: ['accuracy', 'loss', 'val_accuracy', 'val_loss']
```

(For full logs and TensorBoard visuals, run the notebook interactively.)

## Additional Learnings / Reflections

Through this project, I gained hands-on experience with callback mechanics, such as using `self.model.stop_training` for custom stopping logic and accessing `logs` for real-time metrics. Explanations in the notebook cover tuple concatenation for input shapes (e.g., `IMAGE_SIZE + (3,)` yielding `(150, 150, 3)`) and math functions like `math.pow` and `math.floor` in schedulers. Some runs with lower accuracies (e.g., ~0.60) were intentional to demonstrate callback interventions, reinforcing my understanding of training pitfalls and optimizations.

## 👤 Author

## Mehran Asgari
## **Email:** [imehranasgari@gmail.com](mailto:imehranasgari@gmail.com).
## **GitHub:** [https://github.com/imehranasgari](https://github.com/imehranasgari).

---

## 📄 License

This project is licensed under the MIT License – see the `LICENSE` file for details.

💡 *Some interactive outputs (e.g., plots, widgets) may not display correctly on GitHub. If so, please view this notebook via [nbviewer.org](https://nbviewer.org) for full rendering.*