from langchain.llms import OpenAI
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

st.header("Chat your data")
st.subheader("Streamlit + ChatGPT + Langchain with `stream=True`")

user_input = st.text_input("You: ",placeholder = "Ask me anything ...", key="input")


loader = UnstructuredURLLoader(["https://en.wikipedia.org/wiki/Eurovision_Song_Contest"])
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
docsearch = Chroma(docs, embeddings)

def gen_prompt(docs, query) -> str:
    return f"""To answer the question please only use the Context given, nothing else. Do not make up answer, simply say 'I don't know' if you are not sure.
Question: {query}
Context: {[doc.page_content for doc in docs]}
Answer:
"""

def prompt(query):
     docs = docsearch.similarity_search(query, k=4)
     prompt = gen_prompt(docs, query)
     return prompt

if st.button("Submit", type="primary"):
    st.markdown("----")
    res_box = st.empty()
    report = []
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", 
                                             messages=[
          {"role": "system", "content": "You're an assistant."},
          {"role": "user", "content": f"{prompt(user_input)}"},
          ], 
        stream=True, 
        max_tokens=100,
        temperature=0)
    
    for line in completion:
        if 'content' in line['choices'][0]['delta']:
            # join method to concatenate the elements of the list 
            # into a single string, 
            # then strip out any empty strings
            report.append(line['choices'][0]['delta']['content'])
        result = "".join(report).strip()
        result = result.replace("\n", "")
        res_box.markdown(f'*{result}*')
            
st.markdown("----")
