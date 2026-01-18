import streamlit as st
import torch
import torch.nn as nn
from data_utils import load_vocab, basic_english_tokenizer, Vocab
from model import SentimentLSTM
import os

# Page config
st.set_page_config(page_title="IMDB 情感分析器", page_icon="🎬")

st.title("🎬 IMDB 影评情感分析")
st.markdown("""
在下方输入影评以查看它是 **正面** 还是 **负面** 的。
该模型是基于 50,000 条 IMDB 评论训练的双向 LSTM。
""")

# Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_resources():
    # Load Vocab
    if not os.path.exists('vocab.pt'):
        st.error("未找到 vocab.pt。请先运行 src/train.py。")
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
        st.error("未找到 lstm-model.pt。请先运行 src/train.py。")
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
user_input = st.text_area("请输入您的评论:", "This movie was absolutely fantastic! The acting was great.")

if st.button("分析情感"):
    if user_input.strip() == "":
        st.warning("请输入一些文本。")
    else:
        with st.spinner('分析中...'):
            score = predict_sentiment(model, vocab, user_input)
            
        sentiment = "正面" if score >= 0.5 else "负面"
        confidence = score if score >= 0.5 else 1 - score
        
        # Color coding
        color = "green" if sentiment == "正面" else "red"
        
        st.markdown(f"### 结果: <span style='color:{color}'>{sentiment}</span>", unsafe_allow_html=True)
        st.progress(score)
        st.caption(f"置信度分数: {score:.4f} (0=负面, 1=正面)")
        
        # Expander for details
        with st.expander("查看详情"):
            tokenizer = basic_english_tokenizer
            tokens = tokenizer(user_input)
            st.write("分词:", tokens)
            st.write("原始模型输出 (Sigmoid):", score)
