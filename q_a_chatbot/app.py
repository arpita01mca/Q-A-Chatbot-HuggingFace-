import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# -------------------------------
# Prompt Template
# -------------------------------
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Answer clearly, politely, and informatively."),
        ("user", "Question: {question}")
    ]
)

# -------------------------------
# Generate Response Function
# -------------------------------
@st.cache_resource
def create_llm(model_name, temperature=0.7, max_new_tokens=150):
    """
    Create a Hugging Face pipeline and wrap it in a LangChain LLM.
    """
    hf_pipe = pipeline(
        task="text-generation",
        model=model_name,
        temperature=temperature,
        max_new_tokens=max_new_tokens
    )
    return HuggingFacePipeline(pipeline=hf_pipe)

def get_response(question, llm):
    """
    Generate chatbot response using LangChain HuggingFacePipeline.
    """
    # Format the prompt
    formatted_prompt = prompt_template.format_prompt(question=question).to_string()
    
    # Generate response
    answer = llm(formatted_prompt)
    return answer

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

# Initialize LLM
llm = create_llm(model_name, temperature, max_tokens)

# User input
user_input = st.text_input("You:", placeholder="Type your question here...")

if user_input:
    try:
        response = get_response(user_input, llm)
        st.markdown(f"**Bot:** {response}")
    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
