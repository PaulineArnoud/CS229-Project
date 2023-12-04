import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

def create_normalized_confusion_matrix(true_labels, predicted_labels, classes, output_filepath=None):
    """
    Plot a confusion matrix.

    Args:
        y_true: True class labels
        y_pred: Predicted class labels
        classes: List of class labels

    Returns:
        None
    """

    cm = confusion_matrix(true_labels, predicted_labels)

    # Normalize the values
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print("cm: ", cm)

    plt.figure(figsize=(8, 6))
    sns.set()
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    
    if output_filepath:
        plt.savefig(output_filepath, bbox_inches='tight')
    else:
        plt.show()

def get_feature_importances(model, word_dictionary, classes, top_n=10, output_filepath=None):
    """
    Get feature importances (conditional probabilities) assigned by a Naive Bayes model.

    Args:
        model: A trained Naive Bayes model (output of fit_naive_bayes_model).
        word_dictionary: A python dict mapping words to integers.
        classes: List of class labels.

    Returns:
        A dictionary where keys are class labels, and values are lists of tuples.
        Each tuple contains a word and its conditional probability for that class.
        The list is sorted by conditional probabilities in descending order.
    """
    feature_importances = {}
    for label in classes:
        class_probs = model["probs"][:, int(label)]  # Corrected index

        # Create a list of tuples containing word and conditional probability
        word_probabilities = [(word, class_probs[word_idx]) for word, word_idx in word_dictionary.items()]

        # Sort the list by conditional probabilities in descending order
        word_probabilities.sort(key=lambda x: x[1], reverse=True)

        feature_importances[label] = word_probabilities

        num_classes = len(feature_importances)
    
    fig, axes = plt.subplots(nrows=1, ncols=num_classes, figsize=(15, 6))

    for idx, (label, importances) in enumerate(feature_importances.items()):
        top_features = importances[:top_n]
        words, probabilities = zip(*top_features)

        axes[idx].barh(range(len(words)), probabilities, tick_label=words)
        axes[idx].set_xlabel('Conditional Probability')
        axes[idx].set_ylabel('Word')
        axes[idx].set_title(f'Top {top_n} Feature Importances for Class {label}')
        axes[idx].invert_yaxis()  # Invert the y-axis to show the most important features at the top

    plt.tight_layout()

    if output_filepath:
        plt.savefig(output_filepath, bbox_inches='tight')
    else:
        plt.show()