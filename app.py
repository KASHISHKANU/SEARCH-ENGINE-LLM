import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.callbacks import StreamlitCallbackHandler
from langgraph.prebuilt import create_react_agent   # ✅ NEW IMPORT
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Arxiv and Wikipedia tools setup
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun(name="Search")

# Streamlit app UI
st.title("🔎 LangChain - Chat with Search")
"""
In this app, you can chat with a LangChain-powered AI that can search
the web, read from Arxiv, and pull info from Wikipedia.
"""

# Sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

# Display previous messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Take user input
if prompt := st.chat_input(placeholder="What do you want to search for?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Initialize the LLM
    llm = ChatOpenAI(api_key=openai_api_key, model_name="gpt-4o", streaming=True)
    tools = [search, arxiv, wiki]

    # ✅ Create new ReAct-style agent (replaces initialize_agent)
    agent_executor = create_react_agent(llm, tools)

    # Display assistant message container
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        try:
            response = agent_executor.invoke({"input": prompt}, config={"callbacks": [st_cb]})
            st.session_state.messages.append({'role': 'assistant', "content": response["output_text"]})
            st.write(response["output_text"])
        except Exception as e:
            st.error(f"⚠️ Error: {e}")
