from langchain. chat_models import ChatOpenAI
import streamlit as st
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
memory = ConversationBufferMemory()

if ask:
    res_box = st.empty()
    with st.spinner('typing...'):
        report = []
        chat = ChatOpenAI(streaming=True, temperature=0.9)
        prefix = 'Please output as markdown and wrap code blocks with ```'
        query = prefix + user_input
        conversation = ConversationChain(
            llm=chat, 
            memory=ConversationBufferMemory()            
        )
        res = conversation.predict(input=query, callbacks=[handler])
    
st.markdown("----")
