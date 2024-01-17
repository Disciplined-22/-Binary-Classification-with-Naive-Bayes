# Binary Classification with Naive Bayes

This Python script demonstrates a simple binary classification task using synthetic text data and a Naive Bayes classifier.

## Overview

The script generates synthetic text data using scikit-learn's `make_classification` function, converts the features to strings, and splits the dataset into training and testing sets. It then uses the Bag-of-Words representation with `CountVectorizer` to vectorize the text data and trains a Naive Bayes classifier (`MultinomialNB`). The accuracy of the classifier is evaluated using the `accuracy_score` metric.

## Requirements
Make sure you have the required Python packages installed. You can install them using:

```bash
pip install scikit-learn
pip install numpy
```

## How to Run

1. Clone the repository or download the script.
2. Open a terminal and navigate to the script's directory.
3. Run the script using:

    ```bash
    python main.py
    ```

   Replace `main.py` with the actual name of your Python script.

## Additional Notes

- The synthetic text data is generated with 20 features, 10 of which are informative.
- The script uses the Bag-of-Words representation for text vectorization.
- The Naive Bayes classifier is trained using the `MultinomialNB` class.
- The accuracy of the classifier is printed as the evaluation metric.


