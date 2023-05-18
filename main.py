import streamlit as st
from langchain. chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import (
    HumanMessage,
)
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import openai
from typing import Any, Dict, List

st.header("AMA")
st.subheader("Streamlit + ChatGPT + Langchain with `stream=True`")

def get_state():
    if "state" not in st.session_state:
        st.session_state.state = {"memory": ConversationBufferMemory(memory_key="chat_history")}
    return st.session_state.state
state = get_state()

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

prompt = PromptTemplate(
    input_variables=["history", "input"], 
    template='Based on the following chat_history, Please reply to the question in format of markdown. chat_history: {chat_history},\n question: {input}'
)

user_input = st.text_input("You: ",placeholder = "Ask me anything ...", key="input")
ask = st.button('ask',type='primary')
st.markdown("----")

class SimpleStreamlitCallbackHandler(BaseCallbackHandler):
    """ Copied only streaming part from StreamlitCallbackHandler """
    
    def __init__(self) -> None:
        self.tokens_area = st.empty()
        self.tokens_stream = ""
        
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        self.tokens_stream += token
        self.tokens_area.markdown(self.tokens_stream)

handler = SimpleStreamlitCallbackHandler()

if ask:
    res_box = st.empty()
    with st.spinner('typing...'):
        report = []
        chat = ChatOpenAI(streaming=True, temperature=0.9)
        conversation = ConversationChain(
            llm=chat, 
            prompt=prompt,
            memory=memory            
        )
        res = conversation.predict(input=user_input, callbacks=[handler])
    
st.markdown("----")
