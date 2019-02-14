""" Class to create the model and testing reports """
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB


class MLModelHelper:
    """ Class to create the model and testing reports """
    @staticmethod
    def create_model(x_train, y_train):
        """ create the model with training data """
        nb_model = GaussianNB()
        nb_model.fit(x_train, y_train.ravel())
        return nb_model

    @staticmethod
    def model_performance_test(model, features, results):
        """ run a performance test against the model """
        nb_predict_test = model.predict(features)
        return metrics.accuracy_score(results, nb_predict_test)

    @staticmethod
    def generate_confusion_matrix(model, features, results):
        """
        |-----------------------------------------------------------|
        |                 | *Predicted False*  |  *Predicted True*  |
        |-----------------------------------------------------------|
        | *Actual False*  |  True Negative     |   False Positive   |
        |-----------------------------------------------------------|
        | *Actual True*   |  False Negative    |    True Positive   |
        |-----------------------------------------------------------|
        """
        nb_predict_test = model.predict(features)
        return metrics.confusion_matrix(results, nb_predict_test)

    @staticmethod
    def generate_classification_report(model, features, results):
        """ run a classification report and return as text """
        nb_predict_test = model.predict(features)
        return metrics.classification_report(results, nb_predict_test)
