# Mahabharata_V0.1

## Mahabharata Text Generation

This project leverages a neural network model to generate text based on the Mahabharata. Users can train the model from scratch or load a pre-trained model, and then generate text by providing a starting string.

## Features

- Load Mahabharata text files from a specified directory.
- Tokenize the text and create vocabulary mappings.
- Batch creation for training the model.
- Define and train an LSTM model for text generation.
- Save and load trained models.
- Generate text using the trained model with adjustable temperature for diversity in generated text.
- Streamlit UI for easy interaction.

## Requirements

- Python 3.6+
- Streamlit
- PyTorch
- NumPy

## Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/Jeevan-04/Mahabharata_V0.1
    cd mahabharata-text-generation
    ```

2. Install the required packages:

    ```bash
    pip install streamlit torch numpy
    ```

## Usage

1. Ensure that you have a directory named `txtfile` in the project root, containing the Mahabharata text files. Each file should have a `.txt` extension.

2. Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

3. Open your web browser and navigate to `http://localhost:8501` to interact with the app.

## How It Works

### Loading Text Data

The text data from the Mahabharata is loaded from a specified directory. All text files in this directory are read and concatenated into a single string.

### Tokenization

The text is tokenized into words, and vocabulary mappings are created:

- `vocab_to_int`: A dictionary mapping each word to a unique integer.
- `int_to_vocab`: A dictionary mapping each unique integer back to the corresponding word.

### Creating Batches

Batches of the tokenized text are created for training. The text is divided into sequences of a specified length (`seq_length`), and input-output pairs are generated for the LSTM model.

### Model Definition

An LSTM model is defined with embedding layers, LSTM layers, and a fully connected output layer. The model parameters, including vocabulary size, embedding size, and hidden layer size, are configurable.

### Training

Users can choose to train the model from scratch. The training process involves:

- Initializing the hidden state of the LSTM.
- Forward passing batches through the model.
- Computing loss and backpropagating the error.
- Saving the model after each epoch.

### Generating Text

The trained model can generate text based on a provided starting string. Users can adjust the temperature parameter to control the randomness of the generated text.

## Streamlit UI

The Streamlit UI allows users to:

- Train the model from scratch.
- Load a pre-trained model.
- Generate text by providing a starting string and adjusting the temperature.

### Example

1. Enter the path to the directory containing the Mahabharata text files.
2. Optionally train the model by selecting the "Train model from scratch" checkbox.
3. Enter the path to a pre-trained model if not training from scratch.
4. Provide a starting string and adjust the temperature.
5. Click "Get Answer" to generate and display the text.

## Acknowledgments

This project uses PyTorch for neural network implementation and Streamlit for building the web application interface.
