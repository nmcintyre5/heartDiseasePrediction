# Heart Disease Prediction using PySpark
This Pyspark script uses a supervised machine learning to predict heart disease using Apache Spark's PySpark library. The model is trained on a dataset containing various factors such as biking, smoking, and other attributes related to heart health.

## Overview

Heart disease is a significant health concern worldwide. This project aims to predict the likelihood of heart disease based on demographic and lifestyle factors. The predictive model is implemented using a linear regression model and leverages the distributed computing capabilities of Apache Spark for scalable data processing.

## Key Features

- Utilizes PySpark's distributed computing for efficient data processing and model training.
- Implements linear regression modeling to predict the probability of heart disease.
- Conducts correlation analysis to identify relevant features.
- Evaluates model performance using root mean squared error (RMSE) and R-squared metrics.

## How to Install

1. Clone the repository:
   ```bash
   git clone https://github.com/nmcintyre5/heartDiseasePrediction.git
    ```
2. Navigate to the project directory:
    ```
    cd heartDiseasePrediction
    ```
3. Install dependencies:
    ```
    pip install pyspark scikit-learn kaggle
    ```

## Credits
This project is based on a Coursera project. 