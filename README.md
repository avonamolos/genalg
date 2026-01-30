# Feature selection with genetic algorithms

This repository contains the implementation and experimental evaluation of a **genetic algorithm (GA) for feature selection** applied to several classification tasks.  
The goal of the project is **dimensionality reduction** while maintaining comparable classification performance.

The project was developed as part of a university course on **Artificial Intelligence**.

---

## Project Overview

In high-dimensional datasets, many features are redundant or noisy. This project explores the use of **genetic algorithms** to identify informative subsets of features and studies their effect on different classifiers.

Key characteristics:
- Feature selection formulated as a **binary optimization problem**
- Each chromosome represents a subset of features
- Fitness is evaluated using **validation accuracy**, optionally penalized by subset size
- Final evaluation is performed on a **held-out test set**

---

## Models Used

Two classifiers with different inductive biases are employed:

- **Logistic Regression**
  - Linear, stable, and interpretable
  - Commonly used in medical and scientific applications

- **K-Nearest Neighbors (KNN)**
  - Non-linear, distance-based classifier
  - Highly sensitive to feature selection and dimensionality

Using both models allows analysis of how feature selection affects classifiers with different assumptions.

---

## 🧬 Genetic Algorithm Details

- **Encoding**: Binary chromosome (1 = feature selected, 0 = feature excluded)
- **Selection**: Roulette wheel selection
- **Crossover**: Uniform crossover
- **Mutation**: Random mutation with a fixed number of mutated genes
- **Elitism**: Best solutions preserved across generations
- **Fitness function**:
  - Classification accuracy on the validation set
  - Optional penalty proportional to the number of selected features

---

## Datasets

The following datasets are used:

- **Breast Cancer Wisconsin**
- **Pima Indians Diabetes**
- **Wine Quality**
- **MADELON**

Each dataset is preprocessed according to its characteristics (e.g. scaling, handling missing values).

---
## How to Run
1. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies
```pip
pip install numpy pandas scikit-learn pygad
```

3. Run an experiment, for example:
```python
python logistic_regression/breast-cancer-ga.py
```

# Evaluation Protocol

Baseline model is trained using all features and evaluated on the validation set
Genetic Algorithm performs feature selection using validation accuracy
Test set is used only once, for final evaluation of the selected feature subset
Accuracy differences are interpreted in the context of dimensionality reduction and model simplicity

### Notes

A small decrease in accuracy after feature selection is considered acceptable if accompanied by a significant reduction in dimensionality
The primary objective is simplification and interpretability, not absolute performance maximization
Results may vary due to the stochastic nature of genetic algorithms

---

# Author

Hristina Solomanova <br>
Faculty of Computer Science and Engineering (FCSE)
