# Deep Learning Approaches for Non-invasive EEG-based Brain-Computer Interface

An EEG motor-imagery classification project comparing a classical CSP–LDA baseline with an EEGNet-style convolutional neural network in PyTorch.

## Project overview

The experiments use the training recordings from BCI Competition IV Dataset 2a. The pipeline processes nine subjects (A01–A09), each with 22 EEG channels and four motor-imagery classes: left hand, right hand, feet, and tongue.

The preprocessing workflow uses MNE to read GDF recordings, mark and remove EOG channels, apply an 8–30 Hz band-pass filter, and extract 0–4 second epochs. The resulting per-subject data have shape `(288 trials, 22 channels, 1001 time points)`.

## Methods

- **CSP–LDA baseline:** Common Spatial Patterns with 10 components followed by Linear Discriminant Analysis. The model is evaluated on a stratified 80/20 subject-specific train/test split (`random_state=42`).
- **EEGNet-style CNN:** A compact PyTorch CNN trained with Adam and cross-entropy loss. A validation split selects the best epoch, and final accuracy is measured on the held-out test split.
- **Analysis:** Subject-level results, confusion matrices, and EEGNet training-loss and validation-accuracy curves are examined to understand performance variation and overfitting.

## Results

| Method | Mean subject-level test accuracy |
| --- | ---: |
| CSP–LDA | **67.05% ± 15.43%** |
| EEGNet-style CNN | **48.70% ± 15.18%** |

Values are the mean and sample standard deviation across the nine subjects from the current experiment. The split seed is fixed; PyTorch model initialization and minibatch shuffling are not fully seeded, so exact EEGNet results can vary between runs. The current results show that CSP–LDA outperformed EEGNet under these preprocessing and training settings. They do not establish that deep learning is unsuitable for EEG. The notebooks also document overfitting behaviour and planned regularisation experiments; no improvement from those experiments is claimed here.

## Repository contents

- `pipeline.ipynb` — subject-level preprocessing, CSP–LDA and EEGNet experiments, and evaluation plots.
- `visualisation.ipynb` — exploratory inspection of the recordings, event labels, filtering, epoching, and baseline model workflow.
- `requirements.txt` — Python package dependencies.

## Reading guide

If you are new to the project, read the notebooks in this order:

1. **Start with `visualisation.ipynb`.** It introduces the GDF recordings and their event annotations, checks the motor-imagery event labels, marks the EOG channels, removes them from the EEG analysis, filters the signal to 8–30 Hz, and creates 0–4 second epochs. It then walks through a single-subject CSP–LDA baseline, a stratified train/test split, a five-fold component-count comparison, and an EEGNet training example. The later cells show confusion matrices and training behaviour.
2. **Then read `pipeline.ipynb`.** This notebook packages the preprocessing and evaluation into functions and applies the subject-specific experiments across A01–A09. It reports per-subject CSP–LDA accuracies and their across-subject mean and sample standard deviation, then trains EEGNet with separate training, validation, and held-out test partitions. Validation accuracy selects the checkpoint; test data are used for the final score. The final cells compare subject results and plot confusion matrices and training/validation curves.

The notebooks are intentionally saved without executed outputs. Run cells from top to bottom in a fresh kernel. Figures and printed scores will appear as the cells execute. Some EEGNet cells train models for multiple subjects and can take considerably longer than the preprocessing and CSP–LDA sections.

## Data and setup

The raw BCI Competition IV 2a GDF recordings are not included. Obtain the dataset separately under its applicable terms and place the training files (`A01T.gdf` through `A09T.gdf`) in `data/BCICIV_2a_gdf/`.

Create an environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Then open Jupyter from the repository root:

```bash
jupyter lab
```

The notebooks expect the local dataset path `data/BCICIV_2a_gdf/`. In `pipeline.ipynb`, the `data_path` variable is set near the top of the notebook; change that variable if your dataset is stored elsewhere. `visualisation.ipynb` starts with the A01 training file and uses the same relative directory.

## How to interpret the evaluation

- The classification task has four labels: left hand, right hand, feet, and tongue. Event labels are encoded as integers for model training.
- The classical pipeline fits CSP using only the training portion of each subject's data, then fits LDA on the CSP features. Its test accuracy is computed on that subject's held-out 20% split.
- EEGNet first holds out a subject-specific test split, then uses a validation subset of the remaining training data to select the best epoch. Its final score is calculated once on the held-out test set.
- The reported mean and standard deviation summarize results across subjects; they are not pooled trial-level accuracy. Subject performance varies substantially, so the mean should be read alongside the per-subject scores and confusion matrices.
- Model initialization and shuffled minibatches add randomness to EEGNet training. The split uses a fixed random state, but the notebook does not fully seed PyTorch, so exact CNN scores may change between runs.
- The regularisation analysis is exploratory. Dropout, weight decay, and early stopping are investigation directions; this repository does not claim that they have improved the reported results.

## Notes

This is an independent experimental project on a small, subject-specific dataset. Results should be interpreted in the context of the stated split protocol and current hyperparameters.
