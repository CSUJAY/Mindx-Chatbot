import streamlit as st
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import datetime  # For logging
import json  # Save chat history

# Load dataset
df = pd.read_csv("Chatbot.csv")

# Filter questions and answers
questions = df[df["name"] == "User"]["line"].tolist()
answers = df[df["name"] == "ECO"]["line"].tolist()

# Load Hugging Face Models
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # For finding similar questions
chatbot_model_name = "facebook/blenderbot-400M-distill"
chatbot_model = AutoModelForSeq2SeqLM.from_pretrained(chatbot_model_name)
chatbot_tokenizer = AutoTokenizer.from_pretrained(chatbot_model_name)

# Generate embeddings for dataset questions
question_embeddings = embedding_model.encode(questions, convert_to_tensor=True)

def get_best_response(user_input):
    """Finds the closest matching dataset question using sentence similarity or generates a response."""

    # Compute similarity with dataset questions
    input_embedding = embedding_model.encode(user_input, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(input_embedding, question_embeddings)[0]
    
    best_match_idx = torch.argmax(similarities).item()
    best_match_score = similarities[best_match_idx].item()
    
    # If similarity is high, return dataset answer
    if best_match_score > 0.7:  # Adjust threshold as needed
        return answers[best_match_idx]

    # Otherwise, generate response using BlenderBot
    inputs = chatbot_tokenizer(user_input, return_tensors="pt")
    outputs = chatbot_model.generate(**inputs)
    return chatbot_tokenizer.decode(outputs[0], skip_special_tokens=True)

def log_chat(user_input, bot_response):
    """Logs conversation history to a JSON file."""
    log_entry = {
        "timestamp": str(datetime.datetime.now()),
        "user_input": user_input,
        "bot_response": bot_response
    }
    with open("chat_log.json", "a") as log_file:
        json.dump(log_entry, log_file)
        log_file.write("\n")

# Streamlit UI
st.title("🤖 AI Chatbot")
st.write("Ask me anything!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# **Clear Chat Button**
if st.button("🗑️ Clear Chat"):
    st.session_state.messages.clear()  # Clears chat history
    st.rerun()  # Refresh app

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# **Better Chat Input**
user_input = st.chat_input("Type your message here...")

if user_input:
    # Get chatbot response
    response = get_best_response(user_input)
    
    # Log conversation
    log_chat(user_input, response)
    
    # Save messages in session state
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Display chatbot response
    with st.chat_message("assistant"):
        st.markdown(response)
