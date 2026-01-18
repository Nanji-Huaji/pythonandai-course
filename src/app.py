import streamlit as st
import torch
import torch.nn as nn
from data_utils import load_vocab, basic_english_tokenizer, Vocab
from model import SentimentLSTM
import os

# Page config
st.set_page_config(page_title="IMDB Sentiment Analyzer", page_icon="🎬")

st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.markdown("""
Type a movie review below to see if it's **Positive** or **Negative**.
The model is a Bi-Directional LSTM trained on 50,000 IMDB reviews.
""")

# Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_resources():
    # Load Vocab
    if not os.path.exists('vocab.pt'):
        st.error("vocab.pt not found. Please run src/train.py first.")
        return None, None
    
    vocab = load_vocab('vocab.pt')
    
    # Model Config (Must match training config)
    INPUT_DIM = len(vocab)
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1
    N_LAYERS = 2
    DROPOUT = 0.5
    PAD_IDX = vocab.pad_index
    
    model = SentimentLSTM(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, DROPOUT, PAD_IDX)
    
    # Load Weights
    if not os.path.exists('lstm-model.pt'):
        st.error("lstm-model.pt not found. Please run src/train.py first.")
        return None, vocab
        
    model.load_state_dict(torch.load('lstm-model.pt', map_location=device))
    model.to(device)
    model.eval()
    
    return model, vocab

model, vocab = load_resources()

def predict_sentiment(model, vocab, sentence):
    if not model or not vocab:
        return 0.5
        
    tokenizer = basic_english_tokenizer
    model.eval()
    tokenized = tokenizer(sentence)
    indexed = [vocab[t] for t in tokenized]
    length = [len(indexed)]
    
    if len(length) == 0 or length[0] == 0:
        return 0.5 
        
    tensor = torch.LongTensor(indexed).unsqueeze(0).to(device)
    length_tensor = torch.LongTensor(length)
    
    with torch.no_grad():
        prediction = torch.sigmoid(model(tensor, length_tensor))
    return prediction.item()

# User Interface
user_input = st.text_area("Enter your review:", "This movie was absolutely fantastic! The acting was great.")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner('Analyzing...'):
            score = predict_sentiment(model, vocab, user_input)
            
        sentiment = "POSITIVE" if score >= 0.5 else "NEGATIVE"
        confidence = score if score >= 0.5 else 1 - score
        
        # Color coding
        color = "green" if sentiment == "POSITIVE" else "red"
        
        st.markdown(f"### Result: <span style='color:{color}'>{sentiment}</span>", unsafe_allow_html=True)
        st.progress(score)
        st.caption(f"Confidence Score: {score:.4f} (0=Neg, 1=Pos)")
        
        # Expander for details
        with st.expander("See details"):
            tokenizer = basic_english_tokenizer
            tokens = tokenizer(user_input)
            st.write("Tokens:", tokens)
            st.write("Raw Model Output (Sigmoid):", score)
