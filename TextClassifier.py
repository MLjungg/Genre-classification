from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from datetime import datetime
import numpy as np


class TextClassifier:
    def __init__(self, dataFrame, n_gram, minimum_lyric_appearance):
        # Matrix having columns: Genre | Lyrics | Genre_id
        self.data = dataFrame

        # n_gram = 1 => Only unigram, n_gram = 2 => Unigram + bigram etc.
        self.ngram = n_gram

        # Minimum number of lyrics a word needs to appear in, in order to get a weight.
        self.minimum_lyric_appearance = minimum_lyric_appearance

        # Key = Name of model, Value = Object of model
        self.classification_models = {'random_forest':None, "KNeighbors":None, "naive_bayes": None, 'decisionTree':None,'linear_SVC':None,'logistic_regression':None}

    def create_word_document_matrix(self, data1, data2):
        count_vect = CountVectorizer(min_df=self.minimum_lyric_appearance,
                                     ngram_range=(1, self.ngram))
        return count_vect.fit_transform(data1), count_vect.transform(data2)

    def calculate_tf_idf(self, word_document_matrix):
        tfidf_transformer = TfidfTransformer(smooth_idf=True, sublinear_tf = True)
        tf_idf = tfidf_transformer.fit_transform(word_document_matrix)
        return tf_idf

    def divide_test_and_train(self):
        """x = Lyrics, y = Genre"""
        x_train, x_test, y_train, y_test = train_test_split(self.data['Lyrics'], self.data['Genre'],
                                                            test_size=0.10,
                                                            random_state=42)

        # X_train_count returns a document-term matrix (frequency of terms that occurs in each lyric).
        x_train_counts, x_test_counts = self.create_word_document_matrix(x_train, x_test)

        # Transform the above document-term matrix to its tf-idf representation (having a score).
        x_train_tfidf = self.calculate_tf_idf(x_train_counts)

        return x_train_tfidf, x_test_counts, y_train, y_test

    def naive_bayes(self, x_train_tfidf, y_train):
        return MultinomialNB().fit(x_train_tfidf, y_train)

    def decisionTree(self, x_train_tfidf, y_train):
        return DecisionTreeClassifier(
            class_weight={"Rock": 0.65, "Country": 1.4, "Electronic": 1.8, "Folk": 1.8, "Indie": 1.8, "Jazz": 1.8,
                          "R&B": 1.8, "Pop": 1.20}, max_depth=15).fit(x_train_tfidf, y_train)

    def random_forest(self, x_train_tfidf, y_train):
        return RandomForestClassifier(class_weight={"Rock": 0.65, "Country": 1.4, "Electronic": 1.8, "Folk": 1.8, "Indie": 1.8, "Jazz": 1.8,
                          "R&B": 1.8, "Pop": 1.20}, n_estimators=100, max_depth=15).fit(x_train_tfidf, y_train)

    def linear_SVC(self, x_train_tfidf, y_train):
        return LinearSVC(random_state=0, C=100).fit(x_train_tfidf, y_train)

    def KNeighbors(self, x_train_tfidf, y_train):
        return KNeighborsClassifier(n_neighbors=40, metric="euclidean").fit(x_train_tfidf, y_train)


    def logistic_regression(self, x_train_tfidf, y_train):
        return LogisticRegression(solver='saga', multi_class='ovr', fit_intercept=True, tol=0.0001, C=0.1,
                                  max_iter=1000,
                                  random_state=0).fit(x_train_tfidf, y_train)

    def combine_classifiers(self, y_predicted):
        print("Predicting values with: combined_classifiers")
        genres = np.unique(self.data["Genre"]).tolist()
        y_predicted_combined = []
        for i in range(0, len(y_predicted["naive_bayes"])):
            genre_counter = len(genres) * [0]
            counter = 0
            for classifier, predicted_value in y_predicted.items():
                if classifier == "KNeighbors" and predicted_value[i] == "Country":
                    y_predicted_combined.append(predicted_value[i])
                    break
                if classifier == "KNeighbors" and predicted_value[i] == "Folk":
                    y_predicted_combined.append(predicted_value[i])
                    break
                if classifier == "random_forest" and predicted_value[i] == "Hip-Hop":
                    y_predicted_combined.append(predicted_value[i])
                    break
                if classifier == "naive_bayes" and predicted_value[i] == "Pop":
                    y_predicted_combined.append(predicted_value[i])
                    break
                else:
                    counter += 1
                    if classifier == "naive_bayes":
                        genre_counter[genres.index(predicted_value[i])] += 1.5
                    elif classifier == 'decisionTree' or classifier == 'random_forest':
                        genre_counter[genres.index(predicted_value[i])] += 0.5
                    elif classifier == "logistic_regression":
                        genre_counter[genres.index(predicted_value[i])] += 0
                    else:
                        genre_counter[genres.index(predicted_value[i])] += 1

                    if counter == len(y_predicted):
                        y_predicted_combined.append(genres[np.argmax(genre_counter)])

        return y_predicted_combined

    def create_models(self, x_train_tfidf, y_train):
        for model in self.classification_models:
            print("Training model with: " + model)
            model_object = eval("self." + model + "(x_train_tfidf,y_train)")
            print("Done: " + str(datetime.now()))
            self.classification_models[model] = model_object

    def get_predict_values(self, x_test_counts):
        predicted_values = {}
        for model_string, model_object in self.classification_models.items():
            print("Predicting values with: " + model_string)
            predicted_values[model_string] = model_object.predict(x_test_counts)
            print("Done: " + str(datetime.now()))
        return predicted_values
