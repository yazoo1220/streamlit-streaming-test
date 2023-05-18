import streamlit as st
from langchain. chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import (
    HumanMessage,
)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import openai
from typing import Any, Dict, List

st.header("AMA")
st.subheader("Streamlit + ChatGPT + Langchain with `stream=True`")

loader = UnstructuredURLLoader(["https://en.wikipedia.org/wiki/Eurovision_Song_Contest"])
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(docs, embeddings)
retriever = db.as_retriever()
                                  
def get_state(): 
     if "state" not in st.session_state: 
         st.session_state.state = {"memory": ConversationBufferMemory(memory_key="chat_history")} 
     return st.session_state.state 
state = get_state()
st.write(state)
st.write(state['memory'])

prompt = PromptTemplate(
    input_variables=["chat_history","question"], 
    template='Based on the following chat_history, Please reply to the question in format of markdown. history: {chat_history}. question: {question}'
)

user_input = st.text_input("You: ",placeholder = "Ask me anything ...")
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
        qa = ConversationalRetrievalChain.from_llm(
            llm=chat, 
            prompt=prompt,
            retriever=retriever,
            memory=state['memory']            
        )
        res = qa({question:user_input, callbacks:[handler]})
    
st.markdown("----")
