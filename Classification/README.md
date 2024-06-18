# 003 Classification

## Introduction

In this project, we focus on the implementation and optimization of classification models to predict buy and sell signals in the stock and cryptocurrency markets. We employ a combination of machine learning algorithms, including Logistic Regression, Support Vector Classification (SVC), and XGBoost, integrated into a voting classifier to enhance the accuracy of our predictions.

- Logistic Regression: This classification algorithm predicts the probability of a binary outcome (1 or 0). It models the relationship between the dependent binary variable and one or more independent variables by estimating probabilities using a logistic function.

- Support Vector Classification (SVC): SVC is a machine learning algorithm used for classification tasks. It finds the optimal hyperplane that maximizes the margin between different classes. The data points closest to the hyperplane are known as support vectors.

- XGBoost: This is an advanced implementation of gradient-boosting decision trees designed for speed and performance. 

For this analysis, we use historical price data and various technical indicators from Apple Inc. (AAPL) and Bitcoin (BTC-USD) in 1-minute and 5-minute intervals.

## What does this project do?

We utilize the optuna library for hyperparameter optimization, further improving the performance of our models. The ultimate goal is to provide an effective tool for anticipating market movements and maximizing expected returns in financial operations.

## Steps to run the code

```markdown
# Instructions to Run the MicroTrading Project

## Steps for Windows:

1. **Create a virtual environment**:
   ```sh
   python -m venv venv
   ```

2. **Activate the virtual environment**:
   ```sh
   .\venv\Scripts\activate
   ```

3. **Upgrade `pip`**:
   ```sh
   pip install --upgrade pip
   ```

4. **Install dependencies**:
   ```sh
   pip install -r Classification\requirements.txt
   ```

5. **Run the main script**:
   ```sh
   python run Classification\__main__.py
   ```

## Steps for Mac:

1. **Create a virtual environment**:
   ```sh
   python3 -m venv venv
   ```

2. **Activate the virtual environment**:
   ```sh
   source venv/bin/activate
   ```

3. **Upgrade `pip`**:
   ```sh
   pip install --upgrade pip
   ```

4. **Install dependencies**:
   ```sh
   pip install -r classification/requirements.txt
   ```

5. **Run the main script**:
   ```sh
   python run Classification\__main__.py
   ```


