from typing import List

import pandas as pd 


class DataLoader:
    """ class loading data """
    def __init__(
            self,
            data_path: str,
    ):
        super(DataLoader, self).__init__()

        self.data_path = data_path

    def load(
            self,
            training_split: float = 0.8,
            threshold: int = 5050, 
            file_type: str = 'excel',
    ) -> List[pd.DataFrame]:
        """ load data 
        
        :param training_split:
            float, percentage of data to use for training

        :param threshold:
            float, threshold for removing columns with missing values

        :return:
            List[pd.DataFrame], list of training and testing data
        """

        # load data
        if file_type == 'csv':
            data = df = pd.read_csv(self.data_path)
        elif file_type == 'excel':
            data = df = pd.read_excel(self.data_path)
        else:
            raise ValueError('File type not correctly defined')
        #data = data.sample(frac=1).reset_index(drop=True)

        # remove columns with missing values that are more than 70% of the total
        data_nan = data.isna().sum()
        drop_columns = data_nan.index[data_nan > threshold]
        data = data.drop(drop_columns, axis=1)
    
        # split data into training and testing
        train_data = data[:int(len(data) * training_split)]
        test_data = data[int(len(data) * training_split):]

        return [train_data, test_data]
