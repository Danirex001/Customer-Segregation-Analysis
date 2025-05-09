# Customer-Segregation-Analysis
Data Analysis and K-Means Clustering machine learning model
# Customer Segmentation using Clustering

This project performs clustering on a dataset with features such as `Value`, `Quantity`, and `Payment_Terms` using `KMeans`. It includes data preprocessing like encoding categorical values and standardizing numerical features, then visualizes the results in a scatter plot.

## Features

- Selects relevant features from a dataset
- Encodes categorical variable (`Payment_Terms`) using Label Encoding
- Scales numerical features using StandardScaler
- Applies KMeans clustering (with 4 clusters)
- Visualizes clusters with a scatter plot

## Requirements

Install dependencies using pip:

```bash
pip install pandas matplotlib scikit-learn
