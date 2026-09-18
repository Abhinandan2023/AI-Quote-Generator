# AI Quote Generator using LSTM

A category-conditioned quote generation application that generates AI-based quotes using an LSTM neural network, TensorFlow and Streamlit.

## Features

- Category-conditioned quote generation
- Text sequence processing using NLP techniques
- LSTM-based next-word prediction
- Temperature-based sampling for text generation
- Repetition penalty to reduce repeated words
- Control over maximum quote length
- Interactive Streamlit web interface

## Tech Stack

Python · TensorFlow · Keras · NumPy · LSTM · NLP · Streamlit

## Workflow

Category Input → Tokenization → Sequence Generation → LSTM → Next-Word Prediction → Generated Quote

## Project Structure

app.py  
quote_lstm_model_v3.keras  
tokenizer_v3.pkl  
requirements.txt  
README.md  
.gitignore

Live Demo

Streamlit App: https://ai-quote-generator-lstm.streamlit.app/

GitHub

https://github.com/Abhinandan2023/AI-Quote-Generator

Author

Abhinandan Maity

GitHub

## Run Locally

```bash
git clone https://github.com/Abhinandan2023/AI-Quote-Generator.git
cd AI-Quote-Generator
pip install -r requirements.txt
python -m streamlit run app.py
