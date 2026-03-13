import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import string

# Load dataset
with open("shakespeare.txt", "r", encoding="utf-8") as file:
    text = file.read().lower()

# limit dataset size to speed up training
text = text[:500000]

# Remove punctuation
text = text.translate(str.maketrans('', '', string.punctuation))

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

total_words = len(tokenizer.word_index) + 1

# Create input sequences
input_sequences = []

for line in text.split("\n"):
    token_list = tokenizer.texts_to_sequences([line])[0]

    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Maximum sequence length
max_sequence_len = max([len(x) for x in input_sequences])

# Pad sequences
input_sequences = np.array(
    pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')
)

# Split into X and y
X = input_sequences[:, :-1]
y = input_sequences[:, -1]

# Build model
model = Sequential()

model.add(Embedding(total_words, 100, input_length=max_sequence_len - 1))
model.add(LSTM(150))
model.add(Dense(total_words, activation='softmax'))

# use sparse categorical loss
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# Early stopping
early_stop = EarlyStopping(monitor='loss', patience=3)

# Train model
model.fit(
    X,
    y,
    epochs=20,
    batch_size=128,
    callbacks=[early_stop]
)

# Text generation function
def generate_text(seed_text, next_words, max_sequence_len):

    for _ in range(next_words):

        token_list = tokenizer.texts_to_sequences([seed_text])[0]

        token_list = pad_sequences(
            [token_list],
            maxlen=max_sequence_len - 1,
            padding='pre'
        )

        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted)

        output_word = ""

        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                output_word = word
                break

        seed_text += " " + output_word

    return seed_text


# Generate sample text
print(generate_text("once upon", 20, max_sequence_len))
print(generate_text("to be or", 15, max_sequence_len))
print(generate_text("the king", 15, max_sequence_len))  