from typing import List, Union

import pandas as pd

import tensorflow as tf
import tensorflow_decision_forests as tfdf


class Preprocess:
    """ Prprocess data """
    def __init__(
            self, 
    ):
        super(Preprocess, self).__init__()

    @staticmethod
    def encode_categorical(
            df: pd.DataFrame,
            cols: List[str],
        ): 
        """ encode categorical columns """
        for col in cols:
            if col in df.columns:
                df[col] = df[col].astype('category').cat.codes + 1
        return df
    
    @staticmethod
    def scale(
        df: pd.DataFrame,
        cols: List[str],
    ) -> pd.DataFrame:
        """ scale data """
        for col in cols:
            if col in df.columns:
                df[col] = df[col] + 1
        return df
    
    @staticmethod
    def treat_nan(
        df: pd.DataFrame,
        cols: List[str],
    ) -> pd.DataFrame:
        """ treat missing values """
        for col in cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0])
        return df
    
    @staticmethod
    def drop(
        df: pd.DataFrame,
        cols: List[str],
    ) -> pd.DataFrame:
        """ drop columns """
        for col in cols:
            if col in df.columns:
                df = df.drop(col, axis=1)
        return df

    
    @classmethod
    def preprocess(
            self,
            df: pd.DataFrame, 
            label: str,
            drop_cols: List[str] = [],
            scale_cols: List[str] = [],
            categorical_cols: List[str] = [],
            tf_data: bool = True,
    ) -> Union[pd.DataFrame, tf.data.Dataset]:
        
        # drop columns
        df = self.drop(df, drop_cols)

        # remove rows with missing values that are more than 30% of the total
        df = df.dropna(thresh=0.7 * len(df.columns))

        # replace categorical missing value with mode
        df = self.treat_nan(df, categorical_cols)

        # scale data to avoid 0 value operations
        df = self.scale(df, scale_cols)

        # encode categorical columns
        df = self.encode_categorical(df, categorical_cols)

        # replace numerical missing value with mean
        df.fillna(df.mean(), inplace=True)

        # rename label column
        df = df.rename(columns={label: 'label'})

        # convert to tf.data.Dataset
        if tf_data:
            df = tfdf.keras.pd_dataframe_to_tf_dataset(df, label='label')

        return df



