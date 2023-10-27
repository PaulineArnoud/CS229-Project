import collections
import numpy as np
from collections import Counter
import util
import nltk
from nltk.corpus import stopwords, words as nltk_words
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from gensim.parsing.preprocessing import STOPWORDS as gensim_stopwords
from wordcloud import STOPWORDS as wordcloud_stopwords

# Function to download NLTK resources
def download_nltk_resources():
    """
    Download NLTK resources (stopwords and words data) if not already downloaded.
    """
    nltk.download('stopwords')
    nltk.download('words')  # Download the words in the English dictionary


def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """
    words = message.split(" ")
    return [word.lower() for word in words]


def create_dictionary(messages, custom_stopwords):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices for each word that is in
    our vocab (which consists of the 1,000 most commonly found words in 

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages
        custom_stopwords (list): List of custom stopwords to exclude from the dictionary.

    Returns:
        A python dict mapping words to integers.
    """
    word_counts = Counter()
    # Define words in the english dictionary
    english_words = set(nltk_words.words())

    # Define stop words from different sources
    stop_words = set(stopwords.words('english'))
    stop_words.update(ENGLISH_STOP_WORDS)
    stop_words.update(gensim_stopwords)
    stop_words.update(wordcloud_stopwords)

    # Merge the provided list of custom stop words with the combined stop words list, if provided
    if custom_stopwords is not None:
        stop_words.update(custom_stopwords)

    for message in messages:
        words = set(get_words(message))
        words = [word for word in words if word not in stop_words and word in english_words]
        word_counts.update(words)

    min_five = list(word for word, count in word_counts.items() if count >= 5)
    frequent_vocab = dict()
    for i, word in enumerate(min_five):
        frequent_vocab[word] = i
    print(frequent_vocab)
    return frequent_vocab
    # *** END CODE HERE ***


def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    of the vocabulary appears in each message. 
    Each row in the resulting array should correspond to each message 
    and each column should correspond to a word of the vocabulary.

    Use the provided word dictionary to map words to column indices. Ignore words that
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers. (vocab)

    Returns:
        A numpy array marking the words present in each message.
        Where the component (i,j) is the number of occurrences of the
        j-th vocabulary word in the i-th message.
    """
    # *** START CODE HERE ***
    n_messages = len(messages)
    n_words = len(word_dictionary)  # 1722
    freq_array = np.zeros((n_messages, n_words), dtype=int)

    for i, message in enumerate(messages):
        words = get_words(message)
        for word in words:
            if word in word_dictionary:
                freq_array[i, word_dictionary[word]] += 1

    return freq_array
    # *** END CODE HERE ***


def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The labels for that training data

    Returns: The trained model
    """
    # figure out priors & p_x_given_y for each class by building counts table
    # *** START CODE HERE ***
    n_messages, n_words = matrix.shape

    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)

    # Compute priors for each class
    priors = {}
    for label in unique_labels:
        priors[label] = np.mean(labels == label)

    # Initialize an array for counts and probabilities for each class
    counts = np.zeros((n_words, n_classes))
    probs = np.zeros((n_words, n_classes))

    for i in range(n_classes):
        label = unique_labels[i]
        class_matrix = matrix[labels == label]
        class_total = class_matrix.sum()
        class_counts = class_matrix.sum(axis=0)
        
        counts[:, i] = class_counts
        probs[:, i] = (class_counts + 1) / (class_total + n_words)

    return {
        "priors": priors,
        "probs": probs
    }
    # *** END CODE HERE ***

def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array containing the predicted class labels
    """
    # *** START CODE HERE ***
    n_messages, n_words = matrix.shape
    n_classes = len(model["priors"])

    # Calculate the log-likelihood for each class
    log_likelihoods = np.zeros((n_messages, n_classes))
    for i in range(n_classes):
        log_likelihoods[:, i] = np.log(model["probs"][:, i]).dot(matrix.T)

    # Add priors to the log-likelihoods
    for i in range(n_classes):
        log_likelihoods[:, i] += np.log(model["priors"][i])

    # Predict the class with the highest log-likelihood for each message
    predictions = np.argmax(log_likelihoods, axis=1)

    return predictions
    # *** END CODE HERE ***


def get_top_five_naive_bayes_words(model, dictionary):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in part-c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: A list of the top five most indicative words in sorted order with the most indicative first
    """
    # *** START CODE HERE ***
    # vector of log odds
    log_odds = np.log(model["p_x_given_y1"] / model["p_x_given_y0"])
    top_5_indices = np.argsort(log_odds)[-5:][::-1]
    return [word for word, i in dictionary.items() if i in top_5_indices]
    # *** END CODE HERE ***


def main():
    train_messages, train_labels = util.load_spam_dataset('train_data.tsv')
    # val_messages, val_labels = load_data('val_data.tsv')
    # test_messages, test_labels = load_data('test_data.tsv')

    custom_stopwords = ["class", "course", "cs", "course", "professor", "physics", "econ"]
    dictionary = create_dictionary(train_messages, custom_stopwords)

    print('Size of dictionary: ', len(dictionary))

    # util.write_json('dictionary', dictionary)

    # train_matrix = transform_text(train_messages, dictionary)
    # val_matrix = transform_text(val_messages, dictionary)
    # test_matrix = transform_text(test_messages, dictionary)

    # naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)

    # naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

    # np.savetxt('your_data_naive_bayes_predictions', naive_bayes_predictions)

    # naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    # print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))


if __name__ == "__main__":
    main()
