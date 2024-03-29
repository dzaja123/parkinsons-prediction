# Parkinson's Disease Prediction

## Overview
This repository contains code for predicting the severity of Parkinson's Disease using telemonitoring data. 
The project involves data exploration, visualization, and the implementation of custom regression models and a Random Forest model.

## Table of Contents
- [Dataset](#dataset)
- [Setup](#setup)
- [Data Exploration](#data-exploration)
- [Data Visualization](#data-visualization)
- [Custom Regression Model](#custom-regression-model)
- [Decision Tree Model](#decision-tree-model)
- [SVM Model](#svm-model)
- [Random Forest Model](#random-forest-model)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)

## Dataset
The dataset used in this project is fetched from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php). 
It includes features related to Parkinson's telemonitoring and the corresponding target variables.

## Setup
1. Clone the repository:

   ```bash
   git clone https://github.com/dzaja123/parkinsons-prediction.git
   cd parkinsons-prediction
   ```

2. Installation
Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the main script:
    ```bash
    python main.py
    ```

# Data Exploration
The data exploration phase involves the following key activities:

- Fetching the dataset
- Displaying correlation matrices
- Exploring the distribution of features

# Data Visualization
Various visualization functions are implemented to gain insights into the data. 
Visualization techniques include:

- Scatter plots
- Histograms
- Pairplots

# Custom Regression Model
A custom regression model is implemented using TensorFlow. 
The architecture of the model includes multiple dense layers with different activation functions.

# Decision Tree Model
A Decision Tree model is employed to capture and understand non-linear relationships within the dataset. 
Decision Trees are tree-like models that recursively split the data based on feature values, providing clear decision paths.

# SVM Model
A Support Vector Machine (SVM) model is utilized to capture complex relationships between features and target variables. 
SVM is a powerful algorithm for both classification and regression tasks, and it works by finding the optimal hyperplane that best separates the data points.

# Random Forest Model
A Random Forest model is implemented as a benchmark for comparison with the custom regression model. 
Random Forests are an ensemble learning technique that combines multiple decision trees to improve overall performance and robustness.

# Results
The evaluation results for both the custom regression model and the Random Forest model are presented, including key metrics such as:

- Explained variance
- Mean absolute error
- Mean squared error
- R² score

# Usage
You can use the provided code to explore and predict Parkinson's Disease severity. 
Adjustments to the models or additional features can be made based on your specific requirements.

# Contributing
Contributions are welcome! Feel free to open issues, propose enhancements, or submit pull requests.
