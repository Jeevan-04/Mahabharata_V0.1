import streamlit as st
import os
import re
import torch
import torch.nn as nn
from collections import Counter
import numpy as np

# Function to load all text files from a directory
def load_texts_from_directory(directory_path):
    all_text = ""
    try:
        for filename in sorted(os.listdir(directory_path)):
            if filename.endswith(".txt"):
                with open(os.path.join(directory_path, filename), 'r', encoding='utf-8') as file:
                    all_text += file.read() + " "
    except FileNotFoundError as e:
        st.error(f"Error: {e}")
        st.stop()
    return all_text

# Get the directory path of text files (assuming it's in the same directory as the script)
directory_path = "txtfile"  # Update with your directory path

# Load the text data from the specified directory
mahabharata_text = load_texts_from_directory(directory_path)

# Check if the text was loaded successfully
if not mahabharata_text:
    st.error("Failed to load text data. Please check the directory path and try again.")
    st.stop()

st.write("Text data loaded successfully.")

# Tokenize text
def tokenize_text(text):
    words = text.split()
    word_counts = Counter(words)
    vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}
    int_to_vocab = {ii: word for word, ii in vocab_to_int.items()}
    return words, vocab_to_int, int_to_vocab

words, vocab_to_int, int_to_vocab = tokenize_text(mahabharata_text)
int_text = [vocab_to_int[word] for word in words]

st.write("Text tokenized successfully.")

# Create batches
def get_batches(int_text, batch_size, seq_length):
    n_batches = len(int_text) // (batch_size * seq_length)
    int_text = int_text[:n_batches * batch_size * seq_length]
    int_text = np.array(int_text)
    int_text = int_text.reshape((batch_size, -1))
    for n in range(0, int_text.shape[1], seq_length):
        x = int_text[:, n:n+seq_length]
        y = np.zeros_like(x)
        y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
        yield x, y

batch_size = 32  # Reduced for quicker demonstration
seq_length = 100
batches = list(get_batches(int_text, batch_size, seq_length))

st.write("Batches created successfully.")

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.lstm.num_layers, batch_size, self.lstm.hidden_size).zero_().to(device),
                  weight.new(self.lstm.num_layers, batch_size, self.lstm.hidden_size).zero_().to(device))
        return hidden

# Set hyperparameters
vocab_size = len(vocab_to_int) + 1
embed_size = 400
hidden_size = 256
num_layers = 2
lr = 0.001

model = LSTMModel(vocab_size, embed_size, hidden_size, num_layers)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

st.write("Model initialized successfully.")

# Streamlit UI
st.title("Mahabharata Text Generation")

# Check if user wants to train the model
train_model = st.checkbox("Train model from scratch")

# Variable to stop training
stop_training = False

# Stop training button
def stop_button_callback():
    global stop_training
    stop_training = True

if st.button("Stop Training", on_click=stop_button_callback):
    stop_training = True

if train_model:
    # Training the model
    num_epochs = 3  # Reduced for quicker demonstration

    progress_bar = st.progress(0)
    loss_text = st.empty()

    for epoch in range(num_epochs):
        if stop_training:
            st.write("Training stopped.")
            break
        st.write(f'Starting epoch: {epoch + 1}/{num_epochs}')
        hidden = model.init_hidden(batch_size)
        for i, (x, y) in enumerate(batches):
            if stop_training:
                st.write("Training stopped.")
                break
            x = torch.tensor(x, dtype=torch.long).to(device)
            y = torch.tensor(y, dtype=torch.long).to(device)

            hidden = tuple([each.data for each in hidden])

            model.zero_grad()
            output, hidden = model(x, hidden)

            loss = criterion(output.view(batch_size * seq_length, vocab_size), y.view(batch_size * seq_length))
            loss.backward()
            optimizer.step()

            if i % 10 == 0:  # Update every 10 batches
                loss_text.text(f'Batch: {i}/{len(batches)}, Loss: {loss.item()}')
                progress_bar.progress((epoch * len(batches) + i + 1) / (num_epochs * len(batches)))

        # Save the model after each epoch
        torch.save(model.state_dict(), f"model_epoch_{epoch + 1}.pth")
        st.write(f"Model saved after epoch {epoch + 1}")

    if not stop_training:
        st.write("Model trained successfully.")

else:
    # Check if user wants to load a pre-trained model
    model_path = st.text_input("Enter the path to the pre-trained model:", "model_epoch_3.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        st.write("Pre-trained model loaded successfully.")
    else:
        st.error("Pre-trained model not found. Please check the path and try again.")
        st.stop()

# Generate text
def generate(model, start_string, int_to_vocab, vocab_to_int, device, top_k=5, length=100, temperature=1.0):
    model.eval()
    input_text = [vocab_to_int.get(word, 0) for word in start_string.split()]  # Default to 0 if word not found
    input_text = torch.tensor(input_text, dtype=torch.long).unsqueeze(0).to(device)
    hidden = model.init_hidden(1)

    generated_text = start_string.split()

    for _ in range(length):
        output, hidden = model(input_text, hidden)
        output = output.squeeze().cpu().detach().numpy()

        # Apply temperature
        output = output / temperature
        p = np.exp(output) / np.sum(np.exp(output))
        
        top_ch = np.argsort(p)[-top_k:]
        top_p = p[top_ch] / np.sum(p[top_ch])
        
        next_word = np.random.choice(top_ch, p=top_p)
        if next_word >= len(int_to_vocab):  # Ensure the next_word index is within bounds
            next_word = top_ch[0]  # Default to the most probable word if index out of bounds

        input_text = torch.tensor([[next_word]], dtype=torch.long).to(device)
        generated_text.append(int_to_vocab[next_word])

    return ' '.join(generated_text)

start_string = st.text_input("Ask your question:", "In the great battle of Kurukshetra,")
temperature = st.slider("Temperature", min_value=0.1, max_value=2.0, value=1.0, step=0.1)

if st.button("Get Answer"):
    try:
        generated_text = generate(model, start_string, int_to_vocab, vocab_to_int, device, temperature=temperature)
        st.write("Answer:")
        st.write(generated_text)
    except Exception as e:
        st.error(f"An error occurred during text generation: {e}")
