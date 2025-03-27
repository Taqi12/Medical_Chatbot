import streamlit as st
import torch
import pickle
from sentence_transformers import SentenceTransformer, util

# Load the SBERT model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load precomputed question embeddings
question_embeddings = torch.load("question_embeddings.pt")

# Load the dataset
df = pickle.load(open("medquad.pkl", "rb"))

# Function to get answer
def get_answer_bert(user_query):
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, question_embeddings)
    best_match = scores.argmax().item()
    return df.iloc[best_match]['answer']

# Set up Streamlit page
st.set_page_config(page_title="Medical Chatbot", page_icon="💊")

# Sidebar
st.sidebar.title("🩺 Medical Chatbot")
st.sidebar.write("🤖 **AI-powered chatbot** to answer health-related queries.")
st.sidebar.write("💡 **How to use:**")
st.sidebar.write("1️⃣ Enter a **health-related question** in the text box.")
st.sidebar.write("2️⃣ Click **Get Answer** to receive an AI-generated response.")
st.sidebar.write("3️⃣ If needed, **rephrase** your question for better accuracy.")
st.sidebar.write("---")
st.sidebar.write("🚀 **Powered by SBERT Model**")
st.sidebar.write("📌 *For general guidance, consult a medical professional.*")

# Main UI
st.title("🤖 AI-Powered Medical Chatbot")
st.write("💬 **Ask any health-related question and get an instant AI-generated response.**")

# User Input
user_query = st.text_input("📝 Type your medical question here:")

# Button for getting the answer
if st.button("🔍 Get Answer"):
    if user_query:
        with st.spinner("Processing your question... ⏳"):
            answer = get_answer_bert(user_query)
        st.success(f"🩺 **Response:** {answer}")
    else:
        st.warning("⚠️ Please enter a question to get an answer.")

# Footer
st.write("---")
st.write("🔬 **Disclaimer:** This chatbot provides AI-generated responses based on existing medical literature. It is not a substitute for professional medical advice. Consult a doctor for medical concerns.")
st.write("💙 Made with ❤️ by [Taqi Javed]")

