from collections import defaultdict
import matplotlib.pyplot as pyplot
from sklearn import metrics
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from main import textclassifier_run


class Statistics:

    def __init__(self, data):
        self.data = data
        self.genres = np.unique(data["Genre"])

    def plot_frequency_of_genres(self):
        self.data.groupby('Genre').Genre.count().plot.pie(autopct='%1.1f%%')
        pyplot.show()

    def plot_vary_ngram(self, model_accuracies):
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        index = 0
        for model, accuracies in model_accuracies.items():
            index += 1
            pyplot.plot(range(1,len(accuracies)+1), accuracies, "ro", linestyle='dashed', label=model, color=colors[index])
        pyplot.ylabel("Accuracy")
        pyplot.xlabel("ngram")
        pyplot.legend(bbox_to_anchor=(1.1, 1.00))
        pyplot.show()

    def plot_vary_documents(self, model_accuracies, minimum_lyric_appearance, maximum_lyric_appearance, step):
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        index = 0
        for model, accuracies in model_accuracies.items():
            index +=1
            pyplot.plot(range(minimum_lyric_appearance, maximum_lyric_appearance+1, step), accuracies, "ro", linestyle='dashed', label=model, color=colors[index])
        pyplot.ylabel("Accuracy")
        pyplot.xlabel("Min documents")
        pyplot.legend(bbox_to_anchor=(1.1, 1.00))
        pyplot.show()

    def get_KPI_for_each_genre(self, predicted_values, true_value):
        for classifier, predicted_value in predicted_values.items():
            print(classifier)
            print(metrics.classification_report(true_value, predicted_value,
                                                 self.genres))

    def calculate_and_plot_confusion_matrix(self, y_test, predicted_values):
        genres = np.unique(self.data["Genre"])

        for method in predicted_values:
            y_predicted = predicted_values.get(method)
            conf_mat = confusion_matrix(y_test, y_predicted)
            fig, ax = pyplot.subplots(figsize=(9, 9))
            sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=genres, yticklabels=genres,
                        cmap="YlGnBu_r")
            pyplot.ylabel('Actual')
            pyplot.xlabel('Predicted')
            ax.set_title(method)
            pyplot.show()

    def vary_parameters_ngram(self, ngram_min, ngram_max):
        classifier_accuracies = defaultdict(list)
        for i in range(ngram_min, ngram_max):
            y_predicted, y_test = textclassifier_run(self.data, ngram=i, minimum_lyric_appearance=15)
            for classifier, predicted_value in y_predicted.items():
                classifier_accuracies[classifier].append(metrics.accuracy_score(y_test, predicted_value))

        self.plot_vary_ngram(classifier_accuracies)

    def vary_parameters_min_documents(self, minimum_lyric_appearance, maximum_lyric_appearance, step):
        classifier_accuracies = defaultdict(list)
        for i in range(minimum_lyric_appearance, maximum_lyric_appearance+1, step):
            y_predicted, y_test = textclassifier_run(self.data, ngram=1, minimum_lyric_appearance=i)
            for classifier, predicted_value in y_predicted.items():
                classifier_accuracies[classifier].append(metrics.accuracy_score(y_test,predicted_value))

        self.plot_vary_documents(classifier_accuracies, minimum_lyric_appearance, maximum_lyric_appearance, step)