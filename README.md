# TextSentimentLSTM Project
## Table of Contents
- [Project Name](#project-name)
- [About](#about)
- [Prerequisites](#prerequisites)
- [Features](#features)
- [Getting Started & Installation](#getting-started--installation)
- [Usage](#usage)
- [Learning Outcomes](#learning-outcomes)
- [Contributing](#contributing)
- [License](#license)
- [Credits & Acknowledgements](#credits--acknowledgements)
- [Contact](#contact)
## Project Name
Text Sentiment LSTM
## About
This project implements a TensorFlow-based Long Short-Term Memory (LSTM) neural network for sentiment analysis on the IMDB movie reviews dataset. It classifies reviews as positive or negative, including data preprocessing, model training, evaluation, visualization of training metrics, and prediction on custom reviews. The project demonstrates key concepts in natural language processing (NLP) and deep learning, making it ideal for learning about LSTMs and text classification.
## Prerequisites
To run this project you need Python (3.6 or higher) installed on your system.


Additional libraries are required: TensorFlow, NumPy, Matplotlib.


Install them using:


`pip install tensorflow numpy matplotlib`

## Features
This TextSentimentLSTM includes these features:

-Loads and preprocesses the IMDB dataset (25,000 training and 25,000 test reviews)

-Builds and trains an LSTM model for binary sentiment classification

-Achieves test accuracy of approximately 85–90% after 5 epochs

-Visualizes training and validation accuracy/loss over epochs using Matplotlib

-Predicts sentiment for sample test reviews and custom user input

-Supports customizable parameters (e.g., epochs, batch size, vocabulary size, sequence length)


## Getting Started & Installation
Clone the repository to your local machine:

`git clone https://github.com/NickAlvarez20/TextSentimentLSTM.git`

Install the required dependencies:

`pip install tensorflow numpy matplotlib`

## Usage
Run the Python script (imdb_sentiment_lstm.py) from the command line:

`python imdb_sentiment_lstm.py`

This will:

-Load and preprocess the IMDB dataset

-Train the LSTM model for 5 epochs with a batch size of 32

-Evaluate the model on the test set and print test accuracy

-Display a plot of training and validation accuracy

-Predict sentiments for the first 5 test reviews and a custom review

-You can modify parameters like epochs, batch_size, max_words, or max_len in imdb_sentiment_lstm.py to experiment with different configurations.

## Learning Outcomes
This project helped me:

-Learn about LSTM networks for processing sequential text data in NLP tasks

-Implement text preprocessing techniques (tokenization, padding, vocabulary limiting) using TensorFlow

-Build, train, and evaluate deep learning models for binary classification using ai assisted development

-Visualize model performance metrics using Matplotlib

-Apply trained models to predict sentiment on real-world text data

## Contributing
This is primarily a personal learning / portfolio repository, so formal contributions aren’t required. However, if you spot bugs, have project ideas, or want to add improvements, feel free to:
1. Fork the repo
2. Create a feature branch
3. Submit a pull request Please include clear explanations of your changes and test any new code.
## License
This repository is open and free for educational use.
*(If you decide on a specific license later, insert it here — e.g. MIT, Apache 2.0, etc.)*
## Credits & Acknowledgements
This project was created by NickAlvarez20 as part of my journey to learn Python and Artificial Intelligence programming. Check out my other repositories to see more of my work!
## Contact
You can find more of my work at [NickAlvarez20 on GitHub](https://github.com/NickAlvarez20).
