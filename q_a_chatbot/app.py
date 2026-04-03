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
    page_icon="🤖"
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
    ["google/flan-t5-small", "google/flan-t5-base"]
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
    """
    text2text_pipe = pipeline(
        task="text2text-generation",
        model=model_name,
        max_new_tokens=max_new_tokens,
        do_sample=True,      # enable sampling
        temperature=temperature,
        top_p=0.9            # nucleus sampling
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
        "Answer the question fully and clearly.\n"
        "If the question asks for an example, provide one.\n"
        "Do not repeat the question.\n\n"
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
        response = chain.run({"question": user_input})

        # Remove repeated input if the model echoes it
        response = response.replace(user_input, "").strip()

        st.markdown(f"**Answer:** {response}")

    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
