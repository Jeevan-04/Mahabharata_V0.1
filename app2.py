import streamlit as st
import os
import re
import signal
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

model = genai.GenerativeModel("gemini-pro")

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

# Function to get response from Google Gemini with timeout
def get_gemini_response(question, timeout=60):
    def signal_handler(signum, frame):
        raise TimeoutError("Request timed out")
    
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(timeout)
    
    try:
        response = model.generate_content(question)
        return response.text
    except TimeoutError as e:
        st.error(f"Timeout error: {e}")
    finally:
        signal.alarm(0)  # Cancel the alarm

# Streamlit UI
st.set_page_config(page_title="Mahabharata Text Generation with Gemini")

st.title("Mahabharata Text Generation")

start_string = st.text_input("Ask your question:", "In the great battle of Kurukshetra,")

if st.button("Ask the question"):
    context = f"Context from Mahabharata:\n{mahabharata_text}\n\nQuestion: {start_string}\nAnswer:"
    response = get_gemini_response(context)
    st.subheader("The response is:")
    st.write(response)
