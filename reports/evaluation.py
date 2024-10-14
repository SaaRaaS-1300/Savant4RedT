import streamlit as st
from PIL import Image

with open("resource/eval_txt_A.md", encoding="UTF-8") as f:
    markdown_txt_A = f.read()

with open("resource/eval_txt_B.md", encoding="UTF-8") as f:
    markdown_txt_B = f.read()

st.markdown(markdown_txt_A, unsafe_allow_html=True)
st.markdown(markdown_txt_B, unsafe_allow_html=True)

st.markdown("---")
st.image("resource/logo_1.png", use_column_width=True)
