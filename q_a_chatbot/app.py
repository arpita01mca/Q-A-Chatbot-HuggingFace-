import streamlit as st
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

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
        "google/flan-t5-base",
        "google/flan-t5-small"
    ]
)

max_tokens = st.sidebar.slider("Max Tokens", 50, 500, 300)

# -------------------------------
# Load Hugging Face Pipeline
# -------------------------------
@st.cache_resource
def create_hf_pipeline(model_name, max_new_tokens=300):
    """
    Load a text2text-generation pipeline with greedy decoding for accuracy.
    """
    text2text_pipe = pipeline(
        task="text2text-generation",
        model=model_name,
        max_new_tokens=max_new_tokens,
        do_sample=False,      # Greedy decoding for consistent factual answers
        temperature=20       # Ignored when do_sample=False
    )
    llm = HuggingFacePipeline(pipeline=text2text_pipe)
    return llm

llm = create_hf_pipeline(model_name, max_tokens)

# -------------------------------
# Prompt Template
# -------------------------------
prompt = PromptTemplate(
    template=(
        "You are a helpful AI assistant for beginners.\n"
        "Answer clearly and accurately in plain English.\n"
        "Do NOT say AI learns from humans directly.\n"
        "Explain that AI learns patterns from data and can create new content.\n"
        "Always include a simple, real-world example.\n"
        "Keep the answer short (1–2 sentences).\n\n"
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
        raw_response = chain.run({"question": user_input}).strip()

        # Ensure the response ends with a period
        if raw_response and not raw_response.endswith(('.', '!', '?')):
            raw_response += '.'

        # Display chat-style
        st.markdown(f"**You:** {user_input}")
        st.markdown(f"**Answer:** {raw_response}")

    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
