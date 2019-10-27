import string
import nltk
import pandas as pd


class Preprocessor:

    def __init__(self, filename):
        self.filename = filename
        self.genre = []
        self.lyrics = []

    def read_and_process_data(self):
        song_number = 0
        temp_lyric = []

        with open("data/" + self.filename, encoding='utf8', errors='ignore') as f:
            next(f)  # Skip first line of doc
            for line in f:
                if line[0:len(str(song_number))] == str(song_number):
                    if song_number != 0:
                        cleaned_lyric = self.clean_lyric(''.join(temp_lyric))
                        if self.bad_lyric(''.join(cleaned_lyric)):
                            del self.genre[-1]

                        else:
                            self.lyrics.append(cleaned_lyric)

                        temp_lyric.clear()

                    song_number += 1
                    line = line.split(",")

                    self.genre.append(line[4])

                    # Lägg till från lyrics i "index raden"
                    temp_lyric.append(''.join(line[5:]))

                else:
                    temp_lyric.append(line)

            self.lyrics.append(self.clean_lyric(''.join(temp_lyric)))

    def bad_lyric(self, lyric):
        if self.genre[-1] == "Other" or self.genre[-1] == "Not Available" or lyric == "\n" or lyric == "instrumental" or lyric == "INSTRUMENTAL" or lyric == "[instrumental]" or lyric == "(Instrumental)" or lyric == "":
            return True

    def clean_lyric(self, line):
        words = nltk.word_tokenize(line)

        # Lowercase all words
        words = [word.lower() for word in words]

        # Remove special characters
        table = str.maketrans('', '', string.punctuation)
        words = [word.translate(table) for word in words]

        # Remove non alpha
        words = [word for word in words if word.isalpha()]

        # Remove stop words
        try:
            stop_words = set(nltk.corpus.stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords')
            stop_words = set(nltk.corpus.stopwords.words('english'))

        # Additional stopwords that is not included in the nltk set.
        own_stopwords = ["nt", "ai", "gon", "na"]
        for own_stopword in own_stopwords:
            stop_words.add(own_stopword)

        words = [word for word in words if word not in stop_words]

        # Lemmatization
        try:
            words = [nltk.stem.WordNetLemmatizer().lemmatize(word) for word in words]
        except LookupError:
            nltk.download("wordnet")
            words = [nltk.stem.WordNetLemmatizer().lemmatize(word) for word in words]

        return ' '.join(words)

    def write_processed_data_to_file(self):
        filename = "data/processed_" + self.filename
        with open(filename, 'w') as f:
            for genre, lyrics in zip(self.genre, self.lyrics):
                f.write('{}\n'.format(genre))
                f.write('{}\n'.format(lyrics))
                f.write("\n")

        return filename

    @staticmethod
    def create_DataFrame(filename):
        genre, lyrics = Preprocessor.get_genre_lyrics(filename)

        data = pd.DataFrame({"Genre": genre, "Lyrics": lyrics})
        data["Genre_id"] = data["Genre"].factorize()[0]

        return data

    @staticmethod
    def get_genre_lyrics(filename):
        new_song = True
        genre = []
        temp_lyrics = []
        lyrics = []

        with open(filename, encoding='utf8', errors='ignore') as f:
            for line in f:
                if new_song:
                    genre.append(line[:-1])
                    new_song = False

                elif line == "\n":
                    lyrics.append(''.join(temp_lyrics))
                    temp_lyrics.clear()
                    new_song = True

                else:
                    temp_lyrics.append(line[:-1])

        return genre, lyrics