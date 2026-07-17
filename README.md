# 42_dslr - Data Science Logistic Regression

A comprehensive machine learning project implementing logistic regression from scratch to predict Hogwarts houses from student data. This project includes data exploration tools and a multi-class classification model trained without scikit-learn's high-level functions.

## Overview

This project builds a **binary and multi-class logistic regression classifier** to categorize students into one of four Hogwarts houses: Gryffindor, Slytherin, Ravenclaw, and Hufflepuff. Built entirely from scratch, this project serves as an educational exercise to deeply understand logistic regression at its core, without relying on high-level machine learning libraries.

The implementation includes:

- Logistic regression training with stochastic gradient descent and mini-batch learning
- Custom mathematical implementations (sigmoid, gradient descent, weight optimization)
- Data exploration and visualization tools for training data analysis
- Model evaluation and prediction capabilities, reaching **98.1% precision** on the test set

## Features

- **Logistic Regression Implementation**
  - Sigmoid activation function
  - Gradient descent optimization
  - Stochastic and mini-batch learning strategies
  - Multi-class classification (One-vs-Rest approach)
  - Precision evaluation script

- **Data Science Tools**
  - `describe.py` - Statistical summary of dataset features
  - `histogram.py` - Distribution visualization of numerical features
    - Question: Which Hogwarts course has a homogeneous score distribution between all four houses?
  - `pair_plot.py` - Pairwise relationships between features
    - Question: Which features should be used for logistic regression?
  - `scatter_plot.py` - Scatter plots for selected feature pairs
    - Question: What are the two features that are most similar?

- **Model Persistence**
  - Save/load trained weights and biases as JSON
  - Reproducible predictions on test data

## Installation & Setup

### Prerequisites
- Python 3.7+
- All required libraries are listed in `requirements.txt`.

### Setup

1. Clone the repository:
```bash
git clone https://github.com/noanjrd/42_dslr
cd 42_dslr
```

2. Install dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```

## Usage

### Data Exploration

Explore the training dataset before modeling:

```bash
# Generate statistical summary
python describe.py

# View feature distributions
python histogram.py

# Analyze feature relationships
python pair_plot.py

# Create scatter plots
python scatter_plot.py
```

### Training the Model

Three training approaches are available, differing in how they update weights during gradient descent:

| Script | Method | Tradeoff |
|---|---|---|
| `logreg_train.py` | Standard (batch) gradient descent | Most stable convergence, slower per epoch |
| `stochastic_gradient_descent.py` | Stochastic gradient descent | Faster updates, noisier convergence |
| `mini_batch.py` | Mini-batch gradient descent | Balance of speed and stability |

```bash
# Standard gradient descent approach
python logistic_regression/logreg_train.py

# Alternative: Stochastic gradient descent
python logistic_regression/stochastic_gradient_descent.py

# Alternative: Mini-batch learning
python logistic_regression/mini_batch.py
```

Each approach generates `weights_and_bias.json` containing the trained model parameters.

### Making Predictions

Predict house assignments for the test dataset:

```bash
python logistic_regression/logreg_predict.py
```

### Evaluating Precision

Evaluate model precision after training and prediction:

```bash
python logistic_regression/precision.py
```

## Project Structure

```
42_dslr/
├── README.md                          # Project documentation
├── describe.py                        # Statistical summary tool
├── histogram.py                       # Distribution visualization
├── pair_plot.py                       # Pairwise feature analysis
├── scatter_plot.py                    # Feature scatter plots
├── weights_and_bias.json              # Trained model parameters
├── datasets/
│   ├── dataset_train.csv              # Training data
│   └── dataset_test.csv               # Test data
└── logistic_regression/
    ├── logreg_train.py                # Model training script
    ├── logreg_predict.py              # Prediction script
    ├── precision.py                   # Precision evaluation script
    ├── stochastic_gradient_descent.py # SGD implementation
    ├── mini_batch.py                  # Mini-batch learning
    └── utils.py                       # Utility functions
```

## Results & Performance

- **Precision: 98.1%** on the test dataset
- **Model Configuration**
  - Epochs: 2500
  - Learning Rate: 0.01
  - Optimization: Gradient Descent with weight adjustment
  - Classes: 4 Hogwarts houses (One-vs-Rest binary classifiers)

_Note: Configuration parameters may vary slightly depending on whether standard gradient descent, stochastic gradient descent, or mini-batch learning is used._

## License

This project is part of the 42 School curriculum and follows the school's academic policies.