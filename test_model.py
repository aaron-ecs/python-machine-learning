""" Test class to check the accuracy of the model """
import unittest
from datahelper import DataHelper
from ml_model_helper import MLModelHelper


class Test(unittest.TestCase):
    """ Test class to check the accuracy of the model """
    mh = MLModelHelper()
    model = None
    x_train = None
    x_test = None
    y_train = None
    y_test = None

    def setUp(self):
        """ Read the data file and create model with training data """
        data = DataHelper().read_data()
        self.x_train, self.x_test, self.y_train, self.y_test = DataHelper().split_data(data)
        self.model = self.mh.create_model(self.x_train, self.y_train)

    def test_model_performance(self):
        """ Test for more than 70% accuracy """
        assert self.mh.model_performance_test(self.model, self.x_train, self.y_train) > 0.7
        assert self.mh.model_performance_test(self.model, self.x_test, self.y_test) > 0.7

    def test_confusion_matrix(self):
        """ Test the false positive and false negatives are less than 5 """
        confusion_matrix = self.mh.generate_confusion_matrix(self.model, self.x_test, self.y_test)
        assert confusion_matrix[0, 1] < 5
        assert confusion_matrix[1, 0] < 5

    def test_classification_report(self):
        """ Prints the the classification report"""
        class_report = self.mh.generate_classification_report(
            self.model, self.x_train, self.y_train)
        print(class_report)
