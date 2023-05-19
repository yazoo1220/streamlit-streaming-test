import streamlit as st
from langchain. chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import (
    HumanMessage,
)
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import openai
from typing import Any, Dict, List


sticky_css = """
<style>
.sticky-textbox {
    position: sticky;
    top: 20px;
    background-color: white;
    padding: 10px;
    z-index: 1;
}
</style>
"""

sticky_html = """
<div class="sticky-textbox">
    <form>
        <label for="input">Enter your input:</label><br>
        <input type="text" id="input" name="input"><br><br>
        <input type="submit" value="Submit">
    </form>
</div>
"""

# Render the CSS and HTML
st.markdown(sticky_css, unsafe_allow_html=True)
st.markdown(sticky_html, unsafe_allow_html=True)




st.header("AMA")
st.subheader("Streamlit + ChatGPT + Langchain with `stream=True`")
                                  
def get_state(): 
     if "state" not in st.session_state: 
         st.session_state.state = {"memory": ConversationBufferMemory(memory_key="chat_history")} 
     return st.session_state.state 
state = get_state()

st.write(state['memory'].load_memory_variables({}))

prompt = PromptTemplate(
    input_variables=["chat_history","input"], 
    template='Based on the following chat_history, Please reply to the question in format of markdown. history: {chat_history}. question: {input}'
)

# Place the text_input inside the bottom-container div
st.markdown('<div class="bottom-container">', unsafe_allow_html=True)
user_input = st.text_input("You: ",placeholder = "Ask me anything ...")
ask = st.button('ask',type='primary')
st.markdown('</div>', unsafe_allow_html=True)

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

if ask:
    res_box = st.empty()
    with st.spinner('typing...'):
        report = []
        chat = ChatOpenAI(streaming=True, temperature=0.9)
        conversation = ConversationChain(
            llm=chat, 
            prompt=prompt,
            memory=state['memory']            
        )
        st.write("Input:", user_input)
        st.markdown("----")
        handler = SimpleStreamlitCallbackHandler()
        res = conversation.predict(input=user_input, callbacks=[handler])
        user_input = ''
    
st.markdown("----")
