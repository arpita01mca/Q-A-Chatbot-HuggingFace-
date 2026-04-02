import streamlit as st
from transformers import pipeline

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Hugging Face QA Chatbot", page_icon="🤖")
st.title("🤖 QA Chatbot with Hugging Face")

st.markdown(
    """
    Ask any question below and get a helpful response from a Hugging Face model.
    """
)

# Sidebar settings
st.sidebar.header("Settings")
model_name = st.sidebar.selectbox(
    "Select a model",
    ["google/flan-t5-small", "google/flan-t5-base"]
)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
max_tokens = st.sidebar.slider("Max Tokens", 50, 300, 150)

# -------------------------------
# Load the Hugging Face pipeline
# -------------------------------
@st.cache_resource
def load_model(model_name, temperature, max_tokens):
    return pipeline(
        task="text2text-generation",
        model=model_name,
        temperature=temperature,
        max_new_tokens=max_tokens
    )

llm = load_model(model_name, temperature, max_tokens)

# -------------------------------
# User input
# -------------------------------
user_input = st.text_input("You:", placeholder="Type your question here...")

if user_input:
    try:
        # Directly generate an answer
        output = llm(user_input)
        answer = output[0]['generated_text']
        st.markdown(f"**Bot:** {answer}")
    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
