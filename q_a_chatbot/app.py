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
    "Ask any question below and get a helpful response from a Hugging Face model."
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
        "google/flan-t5-large",            # better reasoning
        "tiiuae/falcon-7b-instruct",       # stronger instruction model
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
    Create a Hugging Face text2text-generation pipeline wrapped in LangChain.
    Enables sampling for more natural outputs.
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
        "You are a helpful AI assistant.\n"
        "Answer the question clearly and concisely in plain English.\n"
        "Do NOT repeat the question.\n"
        "If an example is requested, provide a simple real-world example.\n\n"
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
        # Run the model
        response = chain.run({"question": user_input})

        # Remove repeated input if echoed
        response = response.replace(user_input, "").strip()

        # Optional: limit to first 2 sentences
        sentences = response.split(". ")
        if len(sentences) > 2:
            response = ". ".join(sentences[:2]) + "."

        # Display as chat
        st.markdown(f"**You:** {user_input}")
        st.markdown(f"**Answer:** {response}")

    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
