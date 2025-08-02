import tensorflow as tf  # Import TensorFlow for machine learning
from tensorflow.keras import layers, models  # Import Keras layers and models for neural networks
from tensorflow.keras.preprocessing.text import Tokenizer  # Import Tokenizer for converting text to numbers
from tensorflow.keras.preprocessing.sequence import pad_sequences  # Import pad_sequences for making text inputs uniform
import numpy as np  # Import NumPy for array operations
import matplotlib.pyplot as plt  # Import Matplotlib for plotting

# 1. Load and preprocess the IMDB dataset
max_words = 10000  # Limit vocabulary to the 10,000 most common words
max_len = 200  # Limit each review to 200 words
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=max_words)  # Load IMDB dataset: 25,000 training and 25,000 test reviews, labels (0=negative, 1=positive)

# Pad sequences to ensure all reviews have the same length
x_train = pad_sequences(x_train, maxlen=max_len)  # Pad or truncate training reviews to 200 words
x_test = pad_sequences(x_test, maxlen=max_len)  # Pad or truncate test reviews to 200 words

# 2. Build a neural network for sentiment analysis
model = models.Sequential([  # Create a sequential model
    layers.Embedding(max_words, 128, input_length=max_len),  # Convert words to 128-dimensional vectors (embeddings)
    layers.LSTM(64, return_sequences=False),  # Add LSTM layer with 64 units to process sequential text data
    layers.Dense(32, activation='relu'),  # Add dense layer with 32 neurons and ReLU activation
    layers.Dense(1, activation='sigmoid')  # Output layer: 1 neuron with sigmoid for binary classification (0 or 1)
])

# 3. Compile the model
model.compile(optimizer='adam',  # Use Adam optimizer to adjust weights
              loss='binary_crossentropy',  # Use binary crossentropy loss for binary classification
              metrics=['accuracy'])  # Track accuracy (correct predictions)

# 4. Train the model
history = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2)  # Train for 5 epochs, 32 reviews per batch, 20% for validation

# 5. Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)  # Test model on test data
print(f"Test accuracy: {test_acc:.4f}")  # Print test accuracy (e.g., 0.8700)

# 6. Visualize training and validation accuracy
plt.figure(figsize=(8, 6))  # Create a figure for plotting
plt.plot(history.history['accuracy'], label='Training Accuracy')  # Plot training accuracy per epoch
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')  # Plot validation accuracy
plt.title('Training and Validation Accuracy')  # Set plot title
plt.xlabel('Epoch')  # Label x-axis
plt.ylabel('Accuracy')  # Label y-axis
plt.legend()  # Show legend
plt.show()  # Display the plot

# 7. Test predictions on sample test reviews
word_index = tf.keras.datasets.imdb.get_word_index()  # Get the word-to-index mapping
reverse_word_index = {value: key for key, value in word_index.items()}  # Reverse mapping for decoding

def decode_review(encoded_review):  # Function to decode numerical review back to text
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])  # Decode words, offset by 3 for special tokens

# Print and predict for first 5 test reviews
for i in range(5):  # Loop over first 5 test reviews
    sample_review = x_test[i:i+1]  # Select test review i
    prediction = model.predict(sample_review)  # Predict sentiment (0 to 1)
    sentiment = 'Positive' if prediction[0] > 0.5 else 'Negative'  # Classify as positive if > 0.5, else negative
    print(f"\nReview {i}: {decode_review(x_test[i])[:100]}...")  # Print first 100 characters of review
    print(f"Predicted: {sentiment} ({prediction[0][0]:.2f}), Actual: {'Positive' if y_test[i] == 1 else 'Negative'}")

# 8. Test on a custom review
custom_review = "This movie was absolutely fantastic, great acting and thrilling plot!"  # Define a custom review
tokenizer = Tokenizer(num_words=max_words)  # Create a tokenizer
tokenizer.fit_on_texts([custom_review])  # Fit tokenizer on custom review
custom_seq = tokenizer.texts_to_sequences([custom_review])  # Convert text to sequence of numbers
custom_padded = pad_sequences(custom_seq, maxlen=max_len)  # Pad to match model input
custom_pred = model.predict(custom_padded)  # Predict sentiment
custom_sentiment = 'Positive' if custom_pred[0] > 0.5 else 'Negative'  # Classify sentiment
print(f"\nCustom Review: {custom_review}")
print(f"Predicted Sentiment: {custom_sentiment} ({custom_pred[0][0]:.2f})")