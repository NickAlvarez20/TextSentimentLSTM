IMDB-SentimentLSTM
A TensorFlow-based Long Short-Term Memory (LSTM) neural network for sentiment analysis on the IMDB movie review dataset. This project implements a model to classify reviews as positive or negative, with code for data preprocessing, model training, evaluation, visualization of training metrics, and prediction on custom reviews. Ideal for learning about LSTMs and text classification.
Table of Contents

Project Overview
Installation
Usage
Model Architecture
Dataset
Results
Contributing
License

Project Overview
This repository contains a Python script that builds, trains, and evaluates an LSTM-based neural network using TensorFlow to perform sentiment analysis on the IMDB dataset. The model classifies movie reviews as positive (1) or negative (0), with additional functionality to visualize training/validation accuracy and predict sentiments for sample and custom reviews.
Installation

Clone the repository:
git clone https://github.com/your-username/IMDB-SentimentLSTM.git
cd IMDB-SentimentLSTM


Set up a virtual environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install dependencies:
pip install tensorflow numpy matplotlib


Verify setup:Ensure Python 3.6+ is installed, and run python --version to confirm.


Usage

Run the script:
python imdb_sentiment_lstm.py

This will:

Load and preprocess the IMDB dataset (25,000 training and 25,000 test reviews).
Train the LSTM model for 5 epochs.
Evaluate the model on the test set.
Print the test accuracy and predictions for the first 5 test reviews.
Display a plot of training and validation accuracy.
Predict sentiment for a custom review.


Expected output:

Test accuracy (e.g., ~85–90% after 5 epochs).
Decoded text, predicted sentiment, and actual sentiment for the first 5 test reviews.
Predicted sentiment for a custom review.
A plot showing training and validation accuracy over epochs.


Modify the script:Adjust parameters like epochs, batch_size, max_words, or max_len in imdb_sentiment_lstm.py to experiment with different configurations.


Model Architecture
The LSTM model is built using TensorFlow's Keras API and consists of:

Embedding Layer: Converts 10,000 unique words into 128-dimensional vectors, with input sequences of 200 words.
LSTM Layer: 64 units to process sequential text data, capturing long-term dependencies.
Dense Layer: 32 neurons with ReLU activation for feature processing.
Output Layer: 1 neuron with sigmoid activation for binary classification (positive/negative).

The model is compiled with the Adam optimizer and binary crossentropy loss.
Dataset
The IMDB dataset contains 25,000 training and 25,000 test movie reviews, labeled as positive (1) or negative (0). Reviews are preprocessed to include the 10,000 most common words and padded/truncated to 200 words, loaded via tf.keras.datasets.imdb.
Results
After training for 5 epochs with a batch size of 32:

Test Accuracy: Approximately 85–90% (varies slightly due to random initialization).
Predictions: The script outputs predicted sentiments (Positive/Negative) for the first 5 test reviews, with decoded text and confidence scores, and compares them to actual labels.
Visualization: A plot of training and validation accuracy over epochs.
Custom Review: Predicts sentiment for a user-defined review (e.g., "This movie was absolutely fantastic...").

Contributing
Contributions are welcome! Please:

Fork the repository.
Create a new branch (git checkout -b feature-branch).
Commit your changes (git commit -m "Add feature").
Push to the branch (git push origin feature-branch).
Open a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for details.
