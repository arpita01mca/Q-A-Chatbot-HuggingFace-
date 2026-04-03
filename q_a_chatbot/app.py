import streamlit as st
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import re

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Hugging Face QA Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 QA Chatbot with Hugging Face")
st.markdown(
    "Ask any question below and get a helpful, beginner-friendly answer."
)

# -------------------------------
# Sidebar Settings
# -------------------------------
st.sidebar.header("⚙️ Settings")

model_name = st.sidebar.selectbox(
    "Select a model",
    [
        "google/flan-t5-small",
        "google/flan-t5-base",
        "google/flan-t5-large",          
        "tiiuae/falcon-7b-instruct",
    ]
)

temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
max_tokens = st.sidebar.slider("Max Tokens", 50, 300, 150)

# -------------------------------
# Load Hugging Face Pipeline
# -------------------------------
@st.cache_resource
def create_hf_pipeline(model_name, temperature=0.7, max_new_tokens=150):
    """
    Load a text2text-generation pipeline with sampling for more natural outputs.
    """
    text2text_pipe = pipeline(
        task="text2text-generation",
        model=model_name,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
    )
    llm = HuggingFacePipeline(pipeline=text2text_pipe)
    return llm

llm = create_hf_pipeline(model_name, temperature, max_tokens)

# -------------------------------
# Prompt Template
# -------------------------------
prompt = PromptTemplate(
    template=(
        "You are a helpful AI assistant for beginners.\n"
        "Answer the question clearly and concisely in plain English.\n"
        "Do NOT repeat the question.\n"
        "Provide a simple, real-world example if relevant.\n"
        "Keep the answer short, friendly, and informative.\n\n"
        "Question: {question}\n"
        "Answer:"
    ),
    input_variables=["question"]
)

chain = LLMChain(llm=llm, prompt=prompt)

# -------------------------------
# User Input
# -------------------------------
user_input = st.text_input(
    "You:",
    placeholder="Type your question here..."
)

# -------------------------------
# Generate Response
# -------------------------------
if user_input:
    try:
        raw_response = chain.run({"question": user_input})

        # Remove echoed input if model repeats the question
        response = re.sub(re.escape(user_input), "", raw_response, flags=re.IGNORECASE).strip()

        # Limit to first 2 sentences for clarity
        sentences = re.split(r'(?<=[.!?]) +', response)
        response = ' '.join(sentences[:2]).strip()

        # Ensure the response ends with a period
        if response and not response.endswith(('.', '!', '?')):
            response += '.'

        # Display chat-style
        st.markdown(f"**You:** {user_input}")
        st.markdown(f"**Answer:** {response}")

    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
