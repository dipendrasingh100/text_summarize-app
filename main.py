import streamlit as st   
from transformers import pipeline, AutoTokenizer, BartForConditionalGeneration 
import time 

@st.cache(allow_output_mutation=True)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    sum = pipeline("summarization",model=model,tokenizer=tokenizer)
    return sum

summarizer = load_model()
st.title("Text Summarization")

ex = st.selectbox("Examples",options=["Your text","Example1"])
if ex == "Example1":
    para = '''The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, 
    and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. 
    During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. 
    It was the first structure to reach a height of 300 metres. 
    Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.'''
    txt = st.text_area("Summarize",value=para, height=200)
else:
    txt = st.text_area("Summarize", height=200,placeholder="Enter Your text..")

# How should i dynamically change the height of text area 
n_o_w = len(txt.split(" "))
with st.sidebar:
    max = st.slider("max length",min_value=100,max_value=n_o_w,step=10,value=130)
    min = st.slider("min length",min_value=10,max_value=100,step=10,value=30)
button = st.button("Proceed")
# st.write(f"Computation time on cpu : {round}sec")

with st.spinner("Generating Summary.."):
    if button and txt:
        t1 = time.time()
        res = summarizer(txt,max_length=max,min_length=min)
        t2 = time.time()
        st.write(res[0]["summary_text"])
        tp = t2-t1
        st.metric("Summarize Time",value=f"{round(tp,4)} sec")

