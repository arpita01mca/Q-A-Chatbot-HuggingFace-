import streamlit as st
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StrOutputParser
from langchain.chains import LLMChain

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Hugging Face QA Chatbot", page_icon="🤖")
st.title("🤖 QA Chatbot with Hugging Face")
st.markdown("Ask any question below and get a helpful response from a Hugging Face model.")

# Sidebar settings
st.sidebar.header("Settings")
model_name = st.sidebar.selectbox(
    "Select a model",
    ["google/flan-t5-small", "google/flan-t5-base"]
)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
max_tokens = st.sidebar.slider("Max Tokens", 50, 300, 150)

# -------------------------------
# Load Hugging Face Pipeline
# -------------------------------
@st.cache_resource
def load_pipeline(model_name, temperature, max_new_tokens):
    return pipeline(
        task="text2text-generation",  # correct for Flan-T5
        model=model_name,
        temperature=temperature,
        max_new_tokens=max_new_tokens
    )

pipe = load_pipeline(model_name, temperature, max_tokens)

# -------------------------------
# User Input
# -------------------------------
user_input = st.text_input("You:", placeholder="Type your question here...")

if user_input:
    try:
        # Combine system instruction + user question
        prompt_text = f"You are a helpful assistant. Answer the following question clearly and politely:\n{user_input}"
        result = pipe(prompt_text)  # call the Hugging Face pipeline directly
        response = result[0]["generated_text"]
        st.markdown(f"**Bot:** {response}")
    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
