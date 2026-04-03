import streamlit as st
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="Hugging Face QA Chatbot", page_icon="🤖")

# -------------------------------
# Custom CSS (fancy styles)
# -------------------------------
st.markdown("""
<style>
.user-msg {
    background-color: #DCF8C6;
    padding: 10px 15px;
    border-radius: 12px;
    margin: 10px 0;
    text-align: right;
    font-size: 16px;
}
.bot-msg {
    background-color: #F1F0F0;
    padding: 10px 15px;
    border-radius: 12px;
    margin: 10px 0;
    text-align: left;
    font-size: 16px;
}
.title {
    text-align: center;
    font-size: 28px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: gray;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Header
# -------------------------------
st.markdown('<div class="title">🤖 QA Chatbot with Hugging Face</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Ask anything and get a helpful response</div>', unsafe_allow_html=True)

# -------------------------------
# Sidebar settings
# -------------------------------
st.sidebar.header("⚙️ Settings")
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
    text2text_pipe = pipeline(
        task="text2text-generation",
        model=model_name,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        pad_token_id=50256
    )
    llm = HuggingFacePipeline(pipeline=text2text_pipe)
    return llm

llm = create_hf_pipeline(model_name, temperature, max_tokens)

# -------------------------------
# Chat history (optional but nice)
# -------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------------
# User input
# -------------------------------
user_input = st.text_input("Type your question here...")

if user_input:
    try:
        prompt = PromptTemplate(
            template=(
                "You are a helpful assistant.\n"
                "Answer the following question clearly and politely.\n\n"
                "Question: {question}\n"
                "Answer:"
            ),
            input_variables=["question"]
        )

        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run({"question": user_input})

        # Save chat
        st.session_state.history.append(("user", user_input))
        st.session_state.history.append(("bot", response))

    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")

# -------------------------------
# Display chat
# -------------------------------
for role, msg in st.session_state.history:
    if role == "user":
        st.markdown(f'<div class="user-msg">{msg}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-msg">{msg}</div>', unsafe_allow_html=True)
