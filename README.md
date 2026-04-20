# lab8-K-Nearest-Neighbors-


**Name:** Reema AlMulla
**Course:** ARTI308 - Machine Learning

## About the Assignment

This project implements the **K Nearest Neighbors (KNN)** classification algorithm on an artificial, anonymized dataset (`KNN_Project_Data`). The dataset contains 1,000 observations with 10 unlabeled feature columns (`XVPM`, `GWYH`, `TRAT`, `TLLZ`, `IGGA`, `HYKR`, `EDFS`, `GUUB`, `MGJM`, `JHZC`) and a binary `TARGET CLASS` column that we aim to predict.

## Steps Performed

1. **Import libraries** – pandas, seaborn, matplotlib, numpy, and the relevant scikit-learn modules.
2. **Load the dataset** into a pandas DataFrame and inspect its head.
3. **Exploratory Data Analysis (EDA)** – a seaborn `pairplot` colored by `TARGET CLASS` to visualize relationships between features.
4. **Standardize the variables** using `StandardScaler` so that distance-based KNN is not dominated by features on larger scales.
5. **Train/Test split** – 70% train, 30% test (`random_state=101`).
6. **Fit a KNN classifier** with `n_neighbors=1` and evaluate using a confusion matrix and classification report.
7. **Choose an optimal K** using the Elbow Method – loop over K from 1 to 39, record the error rate for each model, and plot Error Rate vs. K.
8. **Retrain with the best K** (K=30) and re-evaluate. The improved model achieves significantly higher precision, recall, and F1-score than K=1.
