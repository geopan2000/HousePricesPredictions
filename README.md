# House Prices - Advanced Regression Techniques

## Project Overview

This project is part of the Kaggle competition "House Prices - Advanced Regression Techniques." The main objective is to predict the final sale price of houses based on various features related to their characteristics. The approach taken in this project includes feature engineering, data preprocessing, and the use of machine learning models to achieve competitive predictions.

## Key Features

- **Data Preprocessing**: Handling missing values, outliers, and scaling of features.
- **Feature Engineering**: Transforming ordinal, nominal, and quantitative features to improve model performance.
- **Modeling**: Multiple regression models are used, including Random Forests, Gradient Boosting, and stacking ensembles to optimize prediction accuracy.
- **XGBoost**: The final model used XGBoost for predictions, with the tree method set to 'gpu_hist' to utilize GPU training. Log-transformed predictions were converted back to the original scale using an exponential transformation.
- **Evaluation**: Models were evaluated using cross-validation and testing on a hold-out dataset to ensure robustness.

## Requirements

- Python 3.x
- Libraries: `numpy`, `pandas`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`

## Running the Project

To run the notebook, make sure to adjust the paths in the script to match your local directory structure. Some specific paths that need to be modified include:

- **Model Path**: Update the `model_path` to point to your local directory where the trained XGBoost model is saved.
- **Test Data Path**: Adjust the path for the test data used during the prediction phase.
- **Output Path**: Modify the output path for saving the final predictions (e.g., `predictions.csv`).
