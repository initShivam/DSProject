LSTM Text Generation Project

Overview

This project implements a text generation model using Long Short-Term Memory (LSTM) networks. The model is trained on a large text dataset and learns language patterns in order to generate new text based on a given seed phrase.

The implementation TensorFlow/Keras and demonstrates the complete workflow of a Generative AI model, including preprocessing, training, and text generation.

Project Objective

The goal of this project is to build a neural network capable of predicting the next word in a sequence and generating coherent text from a starting phrase.

The project covers:

- Text preprocessing
- Tokenization
- Sequence generation
- LSTM model design
- Model training
- Generating new text from seed input

Dataset

The model is trained on Shakespeare's text dataset, which is a large public domain collection of plays and writings.

Dataset source:
Project Gutenberg – Shakespeare Complete Works

Example dataset file:

shakespeare.txt

The dataset is converted to lowercase and punctuation is removed before training.

Technologies Used

- Python
- TensorFlow / Keras
- NumPy

These libraries are used for data preprocessing, neural network construction, and training.

 Model Architecture

The model consists of the following layers:

1. Embedding Layer
   Converts word indices into dense vector representations.

2. LSTM Layer
   Learns sequential dependencies in text.

3. Dense Layer with Softmax Activation
   Predicts the probability of the next word in the sequence.

Model structure:

Embedding Layer
       ↓
LSTM Layer
       ↓
Dense (Softmax)

Data Preprocessing Steps

1. Load the text dataset
2. Convert text to lowercase
3. Remove punctuation
4. Tokenize the text
5. Create n-gram sequences
6. Pad sequences to equal length
7. Split data into input sequences (X) and labels (y)

Example sequence generation:

Input:

to be or

Output:

not

Training

The model is trained using:

Loss Function:

sparse_categorical_crossentropy

Optimizer:

Adam

Early stopping is used to stop training if the model stops improving.

Training parameters:

- Epochs: 20
- Batch size: 128

Text Generation

After training, the model generates text using a seed phrase.

Example seed input:

once upon

Generated output example:

once upon a time there lived a noble king who ruled the land with wisdom

The model repeatedly predicts the next word and appends it to the sequence.

Example Generated Outputs

Seed:

once upon

Generated text:

once upon a time there was a noble king who ruled the land with honor

Seed:

to be or

Generated text:

to be or not to be that is the question of the heart

Seed:

the king

Generated text:

the king of england shall rise again and lead his people

How to Run the Project

1. Install dependencies

pip install tensorflow numpy

2. Place dataset

Download Shakespeare text and place it in the project folder:

shakespeare.txt

3. Run the script

python dsproject.py

The script will train the model and generate text based on seed inputs.

Project Structure

project-folder
│
├── dsproject.py
├── shakespeare.txt
└── README.md

Conclusion

This project demonstrates how Recurrent Neural Networks with LSTM layers can be used for text generation tasks. The model learns language patterns from training data and generates new sequences that resemble the original dataset.

It provides a practical introduction to Generative AI and sequence modeling using deep learning.
