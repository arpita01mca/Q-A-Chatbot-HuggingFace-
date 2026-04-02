import streamlit as st
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

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
# Load Hugging Face pipeline
# -------------------------------
@st.cache_resource
def create_hf_pipeline(model_name, temperature=0.7, max_new_tokens=150):
    """Create a Hugging Face text2text-generation pipeline wrapped in LangChain."""
    text2text_pipe = pipeline(
        task="text2text-generation",  # Flan-T5 works with text2text
        model=model_name,
        temperature=temperature,
        max_new_tokens=max_new_tokens
    )
    # Wrap the HF pipeline so it can be called via LangChain
    llm = HuggingFacePipeline(pipeline=text2text_pipe)
    return llm

llm = create_hf_pipeline(model_name, temperature, max_tokens)

# -------------------------------
# User input
# -------------------------------
user_input = st.text_input("You:", placeholder="Type your question here...")

if user_input:
    try:
        # Define a simple prompt template for LangChain
        prompt = PromptTemplate(
            template="You are a helpful assistant. Answer clearly and politely:\n{question}",
            input_variables=["question"]
        )

        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run({"question": user_input})
        st.markdown(f"**Bot:** {response}")

    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
