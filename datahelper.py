""" Helper class to read and split the datafile """
import pandas as pd
from sklearn.model_selection import train_test_split


class DataHelper:
    """ Helper class to read and split the datafile """
    @staticmethod
    def read_data():
        """ read the file from disk """
        return pd.read_csv("./data/member_history.csv")

    @staticmethod
    def read_first_x_rows(data, number):
        """ Return the first x rows for testing """
        return data.head(number)

    @staticmethod
    def check_file_shape(data):
        """ Return the file rows and columns for testing """
        return data.shape

    @staticmethod
    def check_for_null_values(data):
        """ Return boolean if the data has null values """
        return data.isnull().values.any()

    @staticmethod
    def split_data(data):
        """ Return a split of the data. 30% to be kept for testing the model """
        features = data[
            ['gender',
             'age',
             'salary',
             'membership_length',
             'time_browsing',
             'last_purchase'
            ]
            ].values
        result = data['value'].values
        split_test_size = 0.30
        return train_test_split(features, result, test_size=split_test_size, random_state=42)
