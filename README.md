# 4661-house-prices-regression
Project for CS 4661, where the goal is to find a regression model that best fits this housing data: https://www.kaggle.com/c/house-prices-advanced-regression-techniques
# Contents
This section provides a brief overview of the contents of this repo.
## data
This folder contains all the training data, as well as visualization data.
## notebooks
- `Using data_processor.ipynb` is a notebook that explains how to use `data_processor.py`
- `data_processor.py` is a file that is responsible for performing the feature selection, creating the numerical, categorical datasets, one hot encoding, ordinal encoding, FAMD, and imputation.
- `data_visualization.ipynb` is a notebook that peforms visualizations and analysis of the features, such as correlation.
- `factor_analysis_of_multiple_data.ipynb` is a notebook that explains what FAMD is and how it is applied to our dataset.
- `feature_engineering.ipynb` is a notebook that explains our thought process behind our feature engineering
    - Why we imputed instead of removing columns with NaN values
    - What categorical columns we chose for ordinal encoding
    - What columns we chose as categorical for one hot encoding
- `houseprice.ipynb` is a notebook that contains our first model - a linear regression performed on only a few features. This model serves as our "base model" which we compare all of our other models to.
- `hyperparam_tuning.ipynb` is a notebook that contains our hyperparameter tuning experiments. Here we experiment with different models, regularization values, and training data to find the best data, model, and their parameters.
## reports
This folder contains our progress report and our paper.
