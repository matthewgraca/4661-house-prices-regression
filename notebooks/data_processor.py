import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

class DataProcessor:
    def __init__(self):
        # categorical features that present as numerical
        self.extracted_cat_features = [
            'MSSubClass', 'YearBuilt', 'YearRemodAdd', 'BsmtFullBath', 
            'BsmtHalfBath', 'GarageYrBlt', 'MoSold', 'YrSold'
        ]

        # ordinal features
        self.extracted_ord_features = [
            'LandSlope', 'BsmtQual', 'BsmtCond', 'BsmtFinType1', 'BsmtFinType2', 
            'HeatingQC', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 
            'GarageQual', 'GarageCond', 'PoolQC'
        ]

    '''
    Process the original data, isolating the numerical features.
    - The 'Id' feature is automatically dropped.

    Params:
        df: the dataframe containing the **original** data

    Returns:
        A dataframe with only the numerical features
    '''
    def numerical_data(self, df):
        num_df = df.select_dtypes(exclude='object')
        num_df.drop('Id', axis=1, inplace=True)
        num_df.drop(self.extracted_cat_features, axis=1, inplace=True)
        return num_df
    
    '''
    Process the original data, isolating the categorical features.
    - This removes the ordinal features by default 

    Params:
        df: The dataframe containing the **original** data
        drop_ordinal: Determines if the ordinal data should be dropped. 
            True by default.

    Returns:
        A dataframe with only the categorical features
    '''
    def categorical_data(self, df, drop_ordinal=True):
        cat_df = df.select_dtypes(include='object')
        cat_df = pd.concat([cat_df, df[self.extracted_cat_features]], axis=1)
        if drop_ordinal:
            cat_df.drop(self.extracted_ord_features, axis=1, inplace=True)
        return cat_df

    '''
    Process the original data, isolating the ordinal features.
    - If you want only numerical and categorical features, perform ordinal 
        encoding and concatenate to the numerical dataframe.

    Params:
        df: the dataframe containing the **original** data

    Returns:
        A dataframe with only the ordinal features
    '''
    def ordinal_data(self, df):
        return df[self.extracted_ord_features]

    '''
    Process the original training data, into one more usable by regression.
    - All numerical features preserved with NaNs converted to 0.
    - All categorical features one-hot encoded.
    - Categorical features with natural ordering encoded with ordinal encoding.

    Params
        df: the dataframe containing the **original** training data

    Returns
        A dataframe with the above operations performed, usuable for regression.
    '''
    def encode_data(self, df):
        # separate into numerical and categorical
        num_df = self.numerical_data(df) 
        cat_df = self.categorical_data(df)
        ord_df = self.ordinal_data(df)

        # impute numerical NaN values with 0
        num_df.fillna(0, inplace=True)

        # encode categorical features
        enc = OneHotEncoder(sparse_output=False)
        enc.set_output(transform='pandas')
        encoded_cat_df = enc.fit_transform(cat_df)

        # encode ordinal features
        enc = OrdinalEncoder(encoded_missing_value=-1)
        enc.set_output(transform='pandas')
        encoded_ord_df = enc.fit_transform(ord_df)

        # merge all the dfs
        target_col = num_df["SalePrice"]
        num_df.drop("SalePrice", axis=1, inplace=True)
        train_df = pd.concat([num_df, encoded_cat_df, encoded_ord_df, target_col], axis=1)

        return train_df
