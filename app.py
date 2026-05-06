import streamlit as st
import os
from dotenv import load_dotenv

from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ---------------- SETUP ---------------- #
st.set_page_config(page_title="Jarvis AI", layout="centered")


# ✅ Check API key early (prevents blank screen)

api_key = st.secrets["groq_api_key"]
if not api_key:
    st.error("❌ GROQ_API_KEY not found. Add it to your .env file.")
    st.stop()

# ✅ Initialize model
llm = ChatGroq(
    groq_api_key=api_key,
    model="llama-3.1-8b-instant"
)

# ✅ Prompt with memory
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert software engineer who helps with coding  and your name is jarvis bhai"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm

# ---------------- MEMORY ---------------- #
store = {}

def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# ---------------- UI ---------------- #
st.title("🤖 Jarvis Software Engineer")

if "session_id" not in st.session_state:
    st.session_state.session_id = "default"

if "messages" not in st.session_state:
    st.session_state.messages = []

# ✅ Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ✅ Input
user_input = st.chat_input("Ask your code")

if user_input:
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    try:
        # ✅ Safe model call
        response = chain_with_memory.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": st.session_state.session_id}}
        )
        bot_reply = response.content

    except Exception as e:
        bot_reply = f"⚠️ Error: {str(e)}"

    # Show bot reply
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant"):
        st.write(bot_reply)

# ✅ Clear button (proper reset)
if st.button("🗑️ Clear conversation"):
    st.session_state.messages = []
    store.clear()
    st.rerun()
