from langchain. chat_models import ChatOpenAI
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.callbacks.streamlit import StreamlitCallbackHandler
from langchain.schema import (
    HumanMessage,
)
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import openai

st.header("AMA")
st.subheader("Streamlit + ChatGPT + Langchain with `stream=True`")

user_input = st.text_input("You: ",placeholder = "Ask me anything ...", key="input")

class SimpleStreamlitCallbackHandler(StreamlitCallbackHandler):
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """do nothing"""
        pass
    
        def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """do nothing"""
        pass

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """do nothing"""
        pass

handler = SimpleStreamlitCallbackHandler()
memory = ConversationBufferMemory()

if st.button("Submit", type="primary"):
    with st.spinner('typing...'):
        st.markdown("----")
        res_box = st.empty()
        report = []
        chat = ChatOpenAI(streaming=True, temperature=0.9)
        conversation = ConversationChain(
            llm=chat, 
            memory=ConversationBufferMemory()            
        )
        res = conversation.predict(input=user_input, callbacks=[handler])
    

st.markdown("----")
