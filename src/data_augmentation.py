from typing import List, Union

import pandas as pd
import numpy as np

from tqdm import tqdm


class DataAugmentation:
    """ class implementing data augmentation """
    def __init__(
            self, 
    ):  
        super(DataAugmentation, self).__init__()

    @classmethod
    def load(
            self, 
            df: pd.DataFrame,
            label: str,
            opt: str = 'div',
            save: bool = False,
    ) -> pd.DataFrame:
        """ The data are augmented considering the relationship
            between couples of rows/patients

        :param df:
            pd.DataFrame, dataset to augment

        :param label:
            str, label column

        :param opt:
            str, augmentation operation to apply between rows

        :param save:
            bool, whether to save the augmented dataset

        :return:
            pd.DataFrame, augmented dataset
        """

        # parse operation applied between rows
        if opt == 'div':
            opt = lambda x, y: x / y if y != 0 else ""
        elif opt == 'sub':
            opt = lambda x, y: x - y
        else:
            ValueError('Operation for comparison not correctly defined')

        # take couples of rows and apply operation
        new_df = pd.DataFrame()
        for i in tqdm(range(len(df))):
            for j in range(len(df)):
                if i != j:
                    new_row = df.iloc[i] / df.iloc[j]
                    new_df = pd.concat([new_df, pd.DataFrame([new_row])], ignore_index=True)

        # remove infinite values
        new_df = new_df.replace([np.inf, -np.inf], np.nan)

        if save:
            new_df.to_csv('data/augmented_data.csv', index=False)

        return new_df