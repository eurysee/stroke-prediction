# Stroke Prediction and Clustering

This project explores the **Stroke Prediction Dataset** using classical machine learning techniques. It serves as a beginner-level project in the domain of **Artifical Intelligence/Machine Learning**, demonstrating classification, preprocessing, and clustering techniques using real-world medical data.

## Dataset

The dataset used in this project is `stroke.csv` from the [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) on Kaggle. It contains information about patients such as:

- Gender
- Age
- Hypertension
- Heart disease
- Marital status
- Work type
- Residence type
- Average glucose level
- BMI
- Smoking status
- Stroke history (target)

## Objective

- Preprocess and clean the dataset.
- Apply different scaling techniques: **Original**, **MinMaxScaler**, **StandardScaler**.
- Train and evaluate **K-Nearest Neighbors (KNN)** and **Decision Tree (DT)** classifiers on all three datasets.
- Visualize and interpret decision trees.
- Perform **K-Means Clustering** on selected features and evaluate with **Silhouette Score**.

## Technologies & Libraries

- Python 3
- pandas, numpy
- matplotlib
- scikit-learn

## Project Workflow

### 1. Package Import
Basic Python libraries and scikit-learn modules are imported for preprocessing, modeling, evaluation, and visualization.

### 2. Data Preparation
- Categorical columns are label encoded.
- Missing values are imputed using the **median** strategy.
- Three datasets are created:
  - `D1`: Original (imputed only)
  - `D2`: MinMax scaled
  - `D3`: Standard scaled

### 3. Model Training
For each dataset (`D1`, `D2`, `D3`), both **KNN** and **Decision Tree** classifiers are:
- Trained
- Tested
- Evaluated with accuracy and confusion matrix
- DT trees are printed in textual format for interpretability

### 4. Clustering (K-Means)
- K-Means clustering is applied to 2D feature space (`age` vs `avg_glucose_level`)
- Optimal cluster count is evaluated using **Silhouette Score**
- Results are plotted with color-coded clusters
  
### Sample Visualizations
- Decision tree structure (text-based)
- Cluster scatter plots by dataset
- Silhouette scores per cluster count
