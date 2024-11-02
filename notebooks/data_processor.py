import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

'''
Process the orginal training data, into one more usable by regression.
- All numerical features preserved with NaNs converted to 0.
- All categorical features one-hot encoded.
- Categorical features with natural ordering encoded with ordinal encoding.

Params
    df: the dataframe containing the **original** training data

Returns
    A dataframe with the above operations performed, usuable for regression.
'''
def process_for_regression(df):
    # read data
    #df = pd.read_csv("../data/train.csv")

    # separate into numerical and categorical
    num_df = df.select_dtypes(exclude='object')
    cat_df = df.select_dtypes(include='object')

    # 1. prepare numerical df

    # drop: 
    # - useless feature, 
    # - categorical features from numerical df
    # fill nans with 0
    num_df.drop('Id', axis=1, inplace=True)

    extracted_cat_features = ['MSSubClass', 'YearBuilt', 'YearRemodAdd', 'BsmtFullBath', 
                              'BsmtHalfBath', 'GarageYrBlt', 'MoSold', 'YrSold']
    num_df.drop(extracted_cat_features, axis=1, inplace=True)
    num_df.fillna(0, inplace=True)

    # 2. prepare categorical df

    # extract ordinal features
    cat_df = pd.concat([cat_df, df[extracted_cat_features]], axis=1)
    extracted_ord_features = ['LandSlope', 'BsmtQual', 'BsmtCond', 'BsmtFinType1', 'BsmtFinType2', 
                              'HeatingQC', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 
                              'GarageQual', 'GarageCond', 'PoolQC']
    cat_df.drop(extracted_ord_features, axis=1, inplace=True)

    # encode categorical features
    enc = OneHotEncoder(sparse_output=False)
    enc.set_output(transform='pandas')
    encoded_cat_df = enc.fit_transform(cat_df)

    # 3. prepare ordinal df
    ord_df = df[extracted_ord_features]

    # encode ordinal features
    enc = OrdinalEncoder(encoded_missing_value=-1)
    enc.set_output(transform='pandas')
    encoded_ord_df = enc.fit_transform(ord_df)

    # merge all the dfs
    target_col = num_df["SalePrice"]
    num_df.drop("SalePrice", axis=1, inplace=True)
    train_df = pd.concat([num_df, encoded_cat_df, encoded_ord_df, target_col], axis=1)

    return train_df
