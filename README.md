# Linear Classifiers: Logistic Regression & ReLU Classifier

This repository contains implementations of **linear classifiers**:
1. **Logistic Regression** - Uses the sigmoid activation function for binary classification.
2. **ReLU Classifier** - Uses the Rectified Linear Unit (ReLU) activation function.

## How It Works
The user can choose which classifier to use by modifying the variable `classifierChoice` in the script.
- Set `classifierChoice = 'sigmoid'` for **Logistic Regression**.
- Set `classifierChoice = 'relu'` for **ReLU Classifier**.

## Getting Started
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/linear-classifiers.git
   ```
2. Install dependencies (if required):
   ```bash
   pip install numpy matplotlib sklearn
   ```
3. Run the script:
   ```bash
   python main.py
   ```

## Overview of Classifiers
### Logistic Regression (Sigmoid Classifier)
- Uses the **sigmoid function** to map input values to probabilities.
- Suitable for **binary classification** problems.
- Decision boundary is **linear**.

### ReLU Classifier
- Uses the **ReLU (Rectified Linear Unit) function** instead of sigmoid.
- Often used in **deep learning**, but here it is applied as a linear classifier.
- Outputs positive values directly and clips negative values to zero.
