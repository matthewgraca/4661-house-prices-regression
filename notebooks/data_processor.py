import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np

class DataProcessor:
    def __init__(self, df):
        self.df = df

        # categorical features that present as numerical
        self.extracted_cat_features = [
            'MSSubClass', 'YearBuilt', 'YearRemodAdd', 'BsmtFullBath', 
            'BsmtHalfBath', 'GarageYrBlt', 'MoSold', 'YrSold'
        ]

        # ordinal features that present as categorical
        self.extracted_ord_features = [
            'LandSlope', 'BsmtQual', 'BsmtCond', 'BsmtFinType1', 'BsmtFinType2', 
            'HeatingQC', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 
            'GarageQual', 'GarageCond', 'PoolQC'
        ]

    '''
    Process the original data, isolating the numerical features.
    - The 'Id' feature is automatically dropped.
    - NaN values imputed with 0
    - Ordinal features encoded and added
    - Target column placed at the end of the dataframe

    Returns:
        A dataframe with only the numerical features
    '''
    def numerical_data(self):
        # set up numerical dataframe
        num_df = self.df.select_dtypes(exclude='object')
        num_df.drop('Id', axis=1, inplace=True)
        num_df.drop(self.extracted_cat_features, axis=1, inplace=True)
        num_df.fillna(0, inplace=True)
        
        # encode ordinal features
        ord_df = self.df[self.extracted_ord_features]
        enc = OrdinalEncoder(encoded_missing_value=-1)
        enc.set_output(transform='pandas')
        encoded_ord_df = enc.fit_transform(ord_df)

        # training set contains SalePrice; test set does not
        if 'SalePrice' in num_df.columns:
            target = num_df['SalePrice']
            num_df.drop('SalePrice', axis=1, inplace=True)
            return pd.concat([num_df, encoded_ord_df, target], axis=1) 
        else:
            return pd.concat([num_df, encoded_ord_df], axis=1) 
    
    '''
    Process the original data, isolating the categorical features.
    - This removes the ordinal features by default 

    Returns:
        A dataframe with only the categorical features
    '''
    def categorical_data(self):
        cat_df = self.df.select_dtypes(include='object')

        # add categorical features that are not 'object' type
        # drop features that are 'object', but will be converted to numeric
        cat_df = pd.concat([cat_df, self.df[self.extracted_cat_features]], axis=1)
        cat_df.drop(self.extracted_ord_features, axis=1, inplace=True)

        # encode categorical features
        enc = OneHotEncoder(sparse_output=False)
        enc.set_output(transform='pandas')
        encoded_cat_df = enc.fit_transform(cat_df)

        return encoded_cat_df

    '''
    Process the original training data, into one more usable by regression.
    - All numerical features preserved with NaNs converted to 0.
    - All ordinal features encoded.
    - All categorical features one-hot encoded.
    - Categorical features with natural ordering encoded with ordinal encoding.

    Returns
        A dataframe with the above operations performed, usuable for regression.
    '''
    def complete_data(self):
        # separate into numerical and categorical
        num_df = self.numerical_data() 
        cat_df = self.categorical_data()

        # merge all the dfs
        # training set has target column, test set does not.
        if 'SalePrice' in num_df.columns:
            target = num_df["SalePrice"]
            num_df.drop("SalePrice", axis=1, inplace=True)
            return pd.concat([num_df, cat_df, target], axis=1)
        else:
            return pd.concat([num_df, cat_df], axis=1)

    '''
    Processes the original training data using Factor Analysis of Multiple Data.
    - Uses complete_data() before transformed by FAMD.
    - Numerical data is scaled, categorical is divided by modality.

    Returns
        The numerical dataframe and categorical dataframe transformed
    '''
    def famd_data(self):
        # get numerical data
        X_num = self.numerical_data()
        
        # divide each one hot encoded column by its modality
        X_cat = self.categorical_data()
        for col in X_cat.columns:
            n = X_cat[col].sum()
            X_cat[col] = X_cat[col].apply(lambda x: x / np.sqrt(n))

        return X_num, X_cat
