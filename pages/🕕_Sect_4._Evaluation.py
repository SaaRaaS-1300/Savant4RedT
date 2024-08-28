import streamlit as st
from PIL import Image

with open("resource/markdown_txt_begin_3.md", encoding="UTF-8") as f:
    markdown_txt_begin_3 = f.read()

st.markdown(markdown_txt_begin_3, unsafe_allow_html=True)

image_logo = Image.open('resource/pic_usage_3.png')
st.sidebar.image(
    image_logo,
    use_column_width=True
)