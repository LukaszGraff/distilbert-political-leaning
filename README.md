# Exploring Transformer Fine-Tuning for Political Leaning Prediction

This project explores the fine-tuning of transformer models for predicting political leaning based on text data. The raw data used in this project comes from the paper by Chris Emmery, Marilù Miotto, et al. (2024), titled "Sobr: A corpus for stylometry, obfuscation, and bias on Reddit," presented at LREC-COLING. The data should be downloaded and placed into the [`data`](data) folder.

## Setup

### Creating a New Environment

You can create a new environment using either virtualenv or conda.

#### Using virtualenv

1. Create a virtual environment:
    ```sh
    python -m venv env
    ```

2. Activate the virtual environment:
    - On Windows:
        ```sh
        .\env\Scripts\activate
        ```
    - On macOS and Linux:
        ```sh
        source env/bin/activate
        ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

#### Using conda

1. Create a new conda environment:
    ```sh
    conda create --name transformer-env python=3.8
    ```

2. Activate the conda environment:
    ```sh
    conda activate transformer-env
    ```

3. Install the required packages:
    ```sh
    conda install pytorch torchvision torchaudio -c pytorch
    conda install -c huggingface transformers
    conda install -c conda-forge lime scikit-learn datasets tqdm numpy pandas
    ```

### Verifying the Installation

To verify that the installation was successful, you can run the following command to check if all required packages are installed correctly:
```sh
python -c "import torch, transformers, lime, sklearn, datasets, tqdm, numpy, pandas; print('All packages are installed correctly.')"
```

## Running the Project

To replicate all the results, run the `run_all.py` script located at the root of the project:
```sh
python run_all.py
```

## Repository Structure

The code in this repository is organized into three main directories: `preprocessing`, `training`, and `evaluation`.

### Preprocessing

The preprocessing scripts handle tasks such as loading data, chunking posts, removing short posts, downsampling authors, and splitting data into train, validation, and test sets. The main preprocessing script is located at `preprocess.py`.

### Training

The training scripts are responsible for training the models. This includes fine-tuning transformer models like DistilBERT and training baseline models like TF-IDF with Logistic Regression. The main training scripts are located in the `training` directory.

### Evaluation

The evaluation scripts handle the evaluation of the trained models. This includes both quantitative and qualitative evaluation. The main evaluation scripts are located in the `evaluation` directory.

## Features

- **Data Handling**: Scripts for loading and preprocessing raw data.
- **Model Training**: Scripts for training transformer models and baseline models.
- **Model Evaluation**: Scripts for evaluating model performance using various metrics.
- **Results Replication**: A script to replicate all results by running all necessary steps.

By following the instructions in this README, you should be able to set up the environment, run the project, and explore the results of fine-tuning a transformer model for political leaning prediction.