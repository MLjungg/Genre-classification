import sys
from Preprocessor import *
from Statistics import *
from TextClassifier import *
import os
import warnings
from datetime import datetime


def preprocessor_run(filename):
    if not processed_file_exist(filename):
        print("Preprocessing file...")
        print(datetime.now())
        preprocessor = Preprocessor(filename)
        preprocessor.read_and_process_data()
        preprocessor.write_processed_data_to_file()
        print("Preprocessing done and file saved!")
        print(datetime.now())


    data = Preprocessor.create_DataFrame("data/processed_" + filename)

    return data


def textclassifier_run(data, ngram, minimum_lyric_appearance):
    # Use the formatted file to a dataframe consisting of Genre | Lyrics | Genre_Id
    textClassifier = TextClassifier(data, ngram, minimum_lyric_appearance)

    # Divide dataset
    x_train_tfidf, x_test_counts, y_train, y_test = textClassifier.divide_test_and_train()

    #  Create the model-objects
    textClassifier.create_models(x_train_tfidf, y_train)

    # Predict the y-vector of the test-set
    y_predicted = textClassifier.get_predict_values(x_test_counts)

    # Predict the y_vector using combined classifier
    y_predicted["combined_classifier"] = textClassifier.combine_classifiers(y_predicted)

    return y_predicted, y_test

def statistics_run(data, y_predicted, y_test, arguments):
    # Create statistics object
    statistics = Statistics(data)

    # Plot the genre-distribution
    if "plot" in arguments:
        print("Plotting genre distribution")
        statistics.plot_frequency_of_genres()

    print("\n****RESULTS****\n")

    # Compare predicted values with true-values (accuracy)
    for classifier, predicted_value in y_predicted.items():
        print("Accuracy " + classifier + ":")
        print(str(round(100*metrics.accuracy_score(y_test,predicted_value),2)) + "%\n")

    if "kpi" in arguments:
        statistics.get_KPI_for_each_genre(y_predicted, y_test)

    # Create and visualize confusion matrix
    if "cm" in arguments:
        statistics.calculate_and_plot_confusion_matrix(y_test, y_predicted)


def processed_file_exist(filename):
    return os.path.exists("data/processed_" + filename)

if __name__ == "__main__":

    arguments = sys.argv[1:]
    warnings.filterwarnings("ignore")

    filename = "lyrics.txt"
    data = preprocessor_run(filename)

    # För att köra koden med bästa parametrar
    y_predicted, y_test = textclassifier_run(data, ngram=3, minimum_lyric_appearance=15)
    statistics_run(data, y_predicted, y_test, arguments)

    #För att testa output med olika parametrar, kör koden nedan.
    if "vp" in arguments:
        test = Statistics(data)
        test.vary_parameters_ngram(ngram_min=1, ngram_max = 5)
        test.vary_parameters_min_documents(minimum_lyric_appearance=5, maximum_lyric_appearance=205, step=25)