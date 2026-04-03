import streamlit as st
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Hugging Face QA Chatbot", page_icon="🤖")

# -------------------------------
# Custom CSS Styling
# -------------------------------
st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: 32px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: gray;
    margin-bottom: 20px;
}
.user-msg {
    background-color: #DCF8C6;
    padding: 10px 15px;
    border-radius: 12px;
    margin: 10px 0;
    text-align: right;
}
.bot-msg {
    background-color: #F1F0F0;
    padding: 10px 15px;
    border-radius: 12px;
    margin: 10px 0;
    text-align: left;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Header
# -------------------------------
st.markdown('<div class="main-title">🤖 QA Chatbot with Hugging Face</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Ask anything and get a helpful response</div>', unsafe_allow_html=True)

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
    text2text_pipe = pipeline(
        task="text2text-generation",
        model=model_name,
        max_new_tokens=max_new_tokens,
        do_sample=True,          # IMPORTANT: avoids repetition
        temperature=temperature,
        top_p=0.9
    )
    llm = HuggingFacePipeline(pipeline=text2text_pipe)
    return llm


llm = create_hf_pipeline(model_name, temperature, max_tokens)

# -------------------------------
# Prompt Template
# -------------------------------
prompt = PromptTemplate(
    template=(
        "You are an AI assistant.\n"
        "Give a clear and complete answer.\n"
        "Do not repeat the question.\n\n"
        "Question: {question}\n"
        "Answer:"
    ),
    input_variables=["question"]
)

chain = LLMChain(llm=llm, prompt=prompt)

# -------------------------------
# Chat History
# -------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------------
# User Input
# -------------------------------
user_input = st.text_input("Type your question here...")

if user_input:
    try:
        response = chain.run({"question": user_input})

        # Clean repeated text
        response = response.replace(user_input, "").strip()

        # Save chat
        st.session_state.history.append(("user", user_input))
        st.session_state.history.append(("bot", response))

    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")

# -------------------------------
# Display Chat
# -------------------------------
for role, message in st.session_state.history:
    if role == "user":
        st.markdown(f'<div class="user-msg">{message}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-msg">{message}</div>', unsafe_allow_html=True)
