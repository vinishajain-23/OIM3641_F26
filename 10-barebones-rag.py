"""Bare-bones RAG chatbot over a folder of documents, with setup checks and error handling."""

import os
from pathlib import Path

import httpx
import streamlit as st
from dotenv import load_dotenv
from google.genai import errors as genai_errors
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI

DATA_DIR = Path("data/handbook")

load_dotenv()


def get_api_key():
    """Return the Gemini API key from .env, or stop the app with a message on how to fix it."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error(
            "GEMINI_API_KEY wasn't found. Add a line like `GEMINI_API_KEY=your-key-here` "
            "to the .env file in your project folder, then restart the app."
        )
        st.stop()
    return api_key


def check_data_dir():
    """Make sure DATA_DIR exists, is a folder, and has at least one non-hidden file."""
    if not DATA_DIR.exists():
        st.error(
            f"Couldn't find the folder `{DATA_DIR}`. Create it, put your documents in it, "
            "then restart the app."
        )
        st.stop()
    if not DATA_DIR.is_dir():
        st.error(f"`{DATA_DIR}` exists but it's a file, not a folder. It needs to be a folder of documents.")
        st.stop()

    # Ignore hidden files like .DS_Store, since SimpleDirectoryReader skips them too
    files = [f for f in DATA_DIR.iterdir() if f.is_file() and not f.name.startswith(".")]
    if not files:
        st.error(f"The folder `{DATA_DIR}` is empty. Add at least one document, then restart the app.")
        st.stop()


@st.cache_resource
def get_query_engine(api_key):
    """Load and index the documents once, then reuse the same query engine on every rerun."""
    Settings.llm = GoogleGenAI(model="gemini-2.5-flash", api_key=api_key)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    index = VectorStoreIndex.from_documents(documents)
    return index.as_query_engine()


st.title("Bare Bones RAG Chatbot")

# Fail fast: the app can't do anything without a key and documents
api_key = get_api_key()
check_data_dir()

try:
    query_engine = get_query_engine(api_key)
except OSError as e:
    st.error(
        "Couldn't load the embedding model or read one of your files. Check your internet "
        f"connection and make sure the files in `{DATA_DIR}` aren't corrupted. Details: {e}"
    )
    st.stop()
except Exception as e:
    st.error(f"Something went wrong while indexing your documents: {e}")
    st.stop()

# Chat history so earlier questions stay on screen (stretch goal)
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

prompt = st.chat_input("Ask me anything...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Fallback: one failed question shouldn't take down the whole app
    try:
        with st.spinner("Searching..."):
            response = query_engine.query(prompt)
    except genai_errors.ClientError as e:
        if e.code == 429:
            st.error("You've hit Gemini's rate limit. Wait a minute and try again.")
        else:
            st.error(
                f"Gemini rejected the request (error {e.code}). If this keeps happening, "
                "check that your GEMINI_API_KEY is valid."
            )
    except genai_errors.ServerError:
        st.error("Gemini is having trouble right now. Try your question again in a bit.")
    except (httpx.ConnectError, httpx.TimeoutException):
        st.error("Couldn't reach Gemini. Check your internet connection and try again.")
    except Exception as e:
        st.error(f"Something went wrong answering that question: {e}")
    else:
        bot_response = response.response
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        with st.chat_message("assistant"):
            st.write(bot_response)