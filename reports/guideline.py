import streamlit as st
from PIL import Image

with open("resource/guide_txt_A.md", encoding="UTF-8") as f:
    markdown_txt_A = f.read()

st.markdown(markdown_txt_A, unsafe_allow_html=True)

st.markdown("---")
st.image("resource/logo_1.png", use_column_width=True)
