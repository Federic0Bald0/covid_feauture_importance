from typing import List, Union

import pandas as pd
import numpy as np

import tensorflow as tf

from sklearn.model_selection import train_test_split


class Decorrelator:
    """ Decorrelate data """
    def __init__(
            self, 
            hidden: List[int],

            
    ):
        
        super(Decorrelator, self).__init__()
        self.hidden = hidden

    @staticmethod
    def build_model(
        input_size: int,
        output_size: int,
        hidden: List[int],
        activation: str = 'exponential',
    ) -> tf.keras.Model:
        """ build dnn 
        
        :param input_size:
            int, input size

        :param output_size:
            int, output size

        :param hidden:
            list, hidden layer sizes

        :param activation:
            str, activation function
        """

        # build model
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(input_size,)))
        for layer in hidden:
            model.add(tf.keras.layers.Dense(layer, activation=activation))
        model.add(tf.keras.layers.Dense(output_size, activation=activation))

        return model
    
    def fit(
            self, 
            X: Union[np.ndarray, pd.Series],
            ys: pd.DataFrame,
    ) -> None:
        """ decorrelate data 
        
        :param X:
            np.ndarray or pd.Series, input data

        :param ys:
            pd.DataFrame, output data to decorrelate
        """

        self.models = {}
        for col in ys.columns:
            self.models[col] = self.build_model(
                input_size=1,
                output_size=1,
                hidden=self.hidden,
            )
            self.models[col].compile(
                optimizer='adam',
                loss='mse'
            )
            # early stopping
            callbacks = [tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=10,
            )]
            # fit model
            self.models[col].fit(
                X,
                ys[col],
                epochs=100,
                batch_size=32,
                verbose=0,
                callbacks=callbacks,
            )

    def transform(
            self, 
            X: Union[np.ndarray, pd.Series],
            ys: pd.DataFrame,
    ) -> pd.DataFrame:
        """ transform data """

        for col in self.models:
            ys[col] = ys[col] - self.models[col].predict(X)
        return ys
    
    def fit_transform(
            self, 
            X: Union[np.ndarray, pd.Series],
            ys: pd.DataFrame,
    ) -> pd.DataFrame:
        """ fit and transform data """

        self.fit(X, ys)
        return self.transform(X, ys)

        
