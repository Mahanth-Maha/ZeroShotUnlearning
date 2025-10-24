# Zero-Shot Class Unlearning in Deep Computer Vision Models

**Forgetting the Past**: Targeted Unlearning in Pretrained Deep Networks for Computer Vision

Project Work for **Course** : **DS265 Deep Learning for Computer Vision (DLCV 2025), IISc Bangalore** 

<p align="center">
| <a href="https://github.com/Mahanth-Maha/ZeroShotUnlearning#usage"><b>Documentation</b></a> | <a href="https://mahanthyalla.in/blogs/p/zero-shot-unlearning/"><b>Blog</b></a> | <a href="./WhitePaper_ZeroShotUnlearning.pdf"><b>White Paper</b></a> |
</p>

## Project Overview

This project explores and implements methods for **Machine Unlearning**, specifically focusing on **Zero-Shot Class Unlearning** in deep learning models for computer vision tasks. The goal is to enable a pre-trained model to "forget" specific classes it was originally trained on, without requiring access to the original training data (the "zero-shot" constraint). This is increasingly important due to data privacy regulations (like GDPR's "Right to be Forgotten") and the need to remove outdated or sensitive information from deployed models efficiently.

The core approach leverages **class impressions** generated directly from the trained model's parameters. These impressions act as data-free proxies for the classes to be forgotten or retained. An unlearning algorithm, inspired by gradient ascent/descent techniques (like NegGrad+) but adapted for the zero-shot setting using only these impressions, modifies the model's weights to suppress information related to the "forget" classes while preserving knowledge of the "retain" classes.

This repository contains implementations for:
*   Training baseline models (LeNet5, KarpathyNet, AlexNet, ResNet) on relevant datasets (MNIST, CIFAR-10, ImageNet).
*   Training "Golden Standard" models (retrained from scratch without the forget classes) for comparison.
*   Performing zero-shot class unlearning using class impressions derived from the trained models.
*   Evaluating the effectiveness of unlearning through accuracy metrics and class-wise comparisons.

## Project Structure

```
unlearning_project/
├── data/
│   ├── MNIST/                # MNIST dataset files (idx format)
│   │   └── ...
│   ├── cifar-10-batches-py/  # CIFAR-10 dataset files (python batches)
│   │   └── ...
│   └── ImageNet/             # ImageNet dataset files 
│   │   └── ...
├── Learn_LeNet5_Script.py          # Example training script (can be adapted)
├── Learn_KarpathyNet_Script.py     # Example training script (can be adapted)
├── Learn_KarpathyNet_Golden_Script.py # Example golden standard training script
├── Unlearn_LNet_Script.py          # Unlearning script for LeNet5
├── Unlearn_KNet_Script.py          # Unlearning script for KarpathyNet
├── requirements.txt                # Python package dependencies
└── README.md                       # This file
```

## Setup Instructions

### Prerequisites
*   Git
*   Conda (or Miniconda)
*   NVIDIA GPU with CUDA drivers (recommended for training larger models)

### 1. Clone the Repository
```bash
git clone https://github.com/Mahanth-Maha/ZeroShotUnlearning
cd ZeroShotUnlearning
```

### 2. Create Conda Environment
We recommend using Python 3.10 or later. Create a Conda environment using the provided `requirements.txt` file.

```bash
# Create the environment named 'unlearn_env' (or choose your own)
conda create --name unlearn_env python=3.10 -y

# Activate the environment
conda activate unlearn_env

# Install PyTorch with CUDA support (adjust cuda version if needed)
# Check PyTorch website (pytorch.org) for the correct command for your system/CUDA version
# Example for CUDA 11.8:
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

### 3. Download and Prepare Datasets
*   **MNIST:** Download the IDX files (train-images-idx3-ubyte, train-labels-idx1-ubyte, t10k-images-idx3-ubyte, t10k-labels-idx1-ubyte) and place them in `./data/MNIST/`.
*   **CIFAR-10:** Download the Python version batches (data_batch_1 to 5, test_batch, batches.meta) and place them in `./data/cifar-10-batches-py/`.
*   **ImageNet :** Place your `val`, and `test` image files into `./data/ImageNet/`.
    * image net validation set used is from Kaggle, which can be downloaded from python script 
    ```python   
    import kagglehub
    path = kagglehub.dataset_download("titericz/imagenet1k-val")
    print("Path to dataset files:", path)
    ```
    
>  **Note:** The dataset files are not included in this repository due to size constraints. You can download them from their respective sources:

## Usage

### 1. Training Original Models

* LNet, KNet these are trained with the scripts `Learn_LeNet5_Script.py` and `Learn_KarpathyNet_Script.py` respectively. These scripts are designed to train the models on the specified datasets (MNIST, CIFAR-10) and save the trained models in the `saved_models/` directory.

* AlexNet and ResNet are pretrained models. You can use the `torchvision` library to load these models and fine-tune them on your dataset. The training scripts for these models are not provided in this repository, but you can adapt the existing scripts for your needs.

* Unlearning works on any other model as long as you have the model architecture and the dataset ready.

### 2. Training Golden Standard Models (for Comparison)

These models are trained from scratch *without* the classes intended for forgetting. This provides a benchmark for the unlearning process.

*   **Example (KarpathyNet without classes {3, 4, 8}):**
    *   **Note:** The forget classes `{3, 4, 8}` are currently **hardcoded** in `Learn_KarpathyNet_Golden_Script.py`. You need to **edit the script** to change the `Target_classes` set if you want to forget different classes.
    *   Run the specific script:
        ```bash
        python Learn_KarpathyNet_Golden_Script.py -n v1_golden_forget_348 -e 100 # Use a distinct version name
        ```
    *   Adapt or create similar scripts for other models/forget sets as needed. The output directory will be based on the `--net_name` and `-n`/`--version_name` arguments used in the script.

### 3. Performing Zero-Shot Unlearning

The unlearning scripts load a pre-trained model, generate or load class impressions, and then apply the zero-shot unlearning algorithm.

*   **Notebooks (`.ipynb`):**
    *   Files like `Unlearn_LNet.ipynb` and `Unlearn_KNet.ipynb` provide an interactive way to step through the unlearning process, generate class impressions, visualize results, and understand the core logic. Run these using Jupyter Notebook or Jupyter Lab within your activated conda environment.

*   **Scripts (`.py`):**
    *   Files like `Unlearn_LNet_Script.py` and `Unlearn_KNet_Script.py` are designed to run the unlearning process non-interactively.
    *   **Modify the Scripts:** You'll need to **edit these scripts** to specify:
        *   `net_name` and `version_name`: To load the correct pre-trained model from `saved_models`.
        *   `target_classes`: A set of class indices to forget (e.g., `{1, 2, 3, 4}` or `{3, 4, 8}`).
        *   Hyperparameters like `NUM_SAMPLES` for impressions, `LEARN_RATE` for unlearning, `epochs`, etc.
    *   **Run the Script:**
        ```bash
        # Example for LeNet (after editing the script for desired target classes)
        python Unlearn_LNet_Script.py

        # Example for KarpathyNet (after editing the script)
        python Unlearn_KNet_Script.py
        ```

###  **Output:** The scripts will typically:

1.  Load the specified pre-trained model.
2.  Determine the responsive layer (usually the last learnable one before the classifier).
3.  Generate or load class impressions for all classes for that layer (saving them to `unlearn/<model_name>_<version_name>/class_impressions/`).
4.  Create forget/retain datasets *using the impressions*.
5.  Instantiate the `ZeroShotUnlearner`.
6.  Run the unlearning optimization loop (modifying the model *in memory*).
7.  Evaluate the *unlearned* model on the test set (overall, forget classes, retain classes).
8.  Generate comparison plots (Original vs. Unlearned class-wise accuracy).


*   **Note:** The provided unlearning scripts primarily focus on executing the unlearning process and evaluating its effect *immediately*. They might not explicitly save the *state* of the unlearned model to a separate file. The `u_model` variable within the script holds the unlearned state. You could modify the scripts to save `u_model.state_dict()` if needed.

## Evaluation

*   **Training:** The scripts starting with `Learn` (`.py` and `.ipynb`) evaluates on the validation set during training and saves logs/plots. Final evaluation on the test set (using the best model checkpoint) is performed at the end.
*   **Unlearning:** The unlearning scripts starting with `Unlearn` (`.py` and `.ipynb`) perform evaluation after the unlearning process:
    *   Prints overall accuracy.
    *   Prints accuracy specifically on the *forget* classes (should be close to 0%).
    *   Prints accuracy specifically on the *retain* classes (should be close to the original model's accuracy on these classes, or the golden standard model's accuracy).
    *   Generates bar plots comparing class-wise accuracies of the original model and the unlearned model.



### Note : 
*   The unlearning process is computationally intensive and may take time, especially for larger models and datasets. Ensure you have sufficient resources (GPU recommended).


### Time complexity
class impressions generation: is the most time-consuming part of the unlearning process. but it is done only once for each class. The time complexity of generating class impressions is O(n * m), where n is the number of samples in the dataset and m is the number of classes. This is because we need to compute the class impression for each sample in the dataset for each class.

After generating the class impressions, it is straightforward to compute the forget and retain datasets, and no matter what the number of classes is, the time complexity of this step is O(n), where n is the number of samples in the dataset, for unlearning. The unlearning process itself is O(k * n), where k is the number of epochs and n is the number of samples in the dataset. This is because we need to compute the gradients for each sample in the dataset for each epoch.
