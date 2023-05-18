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


# loader = UnstructuredURLLoader(["https://en.wikipedia.org/wiki/Eurovision_Song_Contest"])
# documents = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# docs = text_splitter.split_documents(documents)
# embeddings = OpenAIEmbeddings()
# docsearch = Chroma.from_documents(docs, embeddings)
handler = StreamlitCallbackHandler()
memory = ConversationBufferMemory()

# def gen_prompt(docs, query) -> str:
#     return f"""To answer the question please only use the Context given, nothing else. Do not make up answer, simply say 'I don't know' if you are not sure.
# Question: {query}
# Context: {[doc.page_content for doc in docs]}
# Answer:
# """

# def prompt(query):
#      docs = docsearch.similarity_search(query, k=4)
#      prompt = gen_prompt(docs, query)
#      return prompt

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
