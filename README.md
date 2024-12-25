# Stroke Prediction Models

This project uses six different models to predict the occurrence of strokes based on various input features. The model is implemented in the [notebook file](https://github.com/timid-angel/stroke-prediction-model/blob/master/Stroke%20Prediction.ipynb), with a detailed [report](https://github.com/timid-angel/stroke-prediction-model/blob/master/Report.pdf).

The dataset is imported from Kaggle, a complete overview of the dataset can be found on their [website](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/data).

# Getting Started

To get started with this project, follow the steps below to set up your environment and run the application.

## Prerequisites

Ensure you have the following installed:
- Python 3.7 or higher
- Jupyter Notebook

Although the Linear Regression algorithm is implemented manually using Batch Gradient Descent, the following libraries are necessary to parse the data, display graphs and test the assumptions of Linear Regression.

The following packages are required to run the jupyter file in its entirety.

- `matplotlib` - for plotting graphs

- `pandas` - for parsing, filtering and cleaning the dataset

- `scikit-learn` - for the classifiers and data set splitting function

- `seaborn` - for plotting categorical features

- `imblearn` - for the SMOTE function


These dependencies can be found in `requirements.txt`. To install all of them, simply run the command:
```bash
make install
```


# Usage

The notebook provides step-by-step instructions and code cells for training a machine learning model to predict the occurrence of strokes based on individual health parameters. Below is a brief overview of the notebook's contents. It is important to note that the purpose of this project is not to choose one model that works best for the particular problem, but rather to analyze each model's performance when applied to the problem domain and report any implications based on the results of each metric. Hence, no validation set has been used and 30% of the dataset has been allocated to the testing set.

This project involves data preprocessing, data visualization and analysis, model training, and evaluation. It uses logistic regression and other classification techniques to predict stroke occurrences.

## Data Preprocessing

This section performs the following operations:

- Handling missing values by imputing median values for certain features (e.g., `bmi`).
- Encoding categorical variables using one-hot encoding for `gender`, `work_type`, `Residence_type`, and other non-numeric columns.
- Scaling numerical features (e.g., `age`, `avg_glucose_level`) using standard scaling to improve model performance.

## Data Analysis

This section visualizes the relationships between the features and the target variable (`stroke`) using:

- Histograms and bar charts for categorical features like `gender` and `smoking_status`.
- Boxplots to identify potential outliers in key features such as `bmi` and `avg_glucose_level`.

## Model Training

This section involves training a logistic regression model using the processed dataset. It includes:

- Splitting the data into training (70%) and testing (30%) sets.
- Importing the following algorithms from `scikit-learn`:
    - Logistic Regression
    - Naive Bayes
    - Gaussian Discriminant Analysis (GDA)
    - Support Vector Machine
    - Decision Tree
    - Random Forest
- Training each model on the training set.


## Model Testing and Performance Metrics

The final model is evaluated on the testing data subset. Performance metrics are calculated to ensure the model generalizes well to unseen data.

The performance of the classification model is evaluated using the following metrics:

- **Accuracy**: Measures the proportion of correctly predicted instances.
- **Precision, Recall, and F1-Score**: Provide insights into the model's balance between positive and negative class predictions.
