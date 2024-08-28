import streamlit as st
import base64
from PIL import Image


def generate_response(model, tokenizer, input_text):
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=50, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def create_the_taskbot():
    # setting
    st.set_page_config(  # 设置页面配置
        page_title="Homepage for Proj",  # 设置网页标题
        page_icon="🌻"
    )
    with open("resource/styles.css", encoding="UTF-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    with open("resource/markdown_txt_begin_1.md", encoding="UTF-8") as f:
        markdown_txt_begin_1 = f.read()
    image = Image.open('resource/pic_usage_1.png')

    # outlook - main
    st.title("🌠 Savant4RedT ⌈内容安全⌋ Expert")
    st.subheader('Overview of the Savant4RedT')
    st.markdown(markdown_txt_begin_1, unsafe_allow_html=True)
    st.markdown("---", unsafe_allow_html=True)
    st.image(
        image,
        use_column_width=True
    )
    st.markdown("---\n", unsafe_allow_html=True)

    # 创建一个按钮，并绑定 JavaScript 的跳转功能
    if st.button('🐬 Read for More Details in Github'):
        st.markdown(
            '<meta http-equiv="refresh" content="0; url=https://github.com/SaaRaaS-1300/Savant4RedT">',
            unsafe_allow_html=True
        )
    # sidebar logo
    image_logo = Image.open('resource/pic_usage_3.png')
    st.sidebar.image(
        image_logo,
        use_column_width=True
    )


if __name__ == "__main__":
    # Load the model and tokenizer
    # model, tokenizer = load_the_model(
    #     model_path=MODEL_PATH,
    #     tokenizer_path=MODEL_PATH
    # )

    # Create the chatbot interface
    create_the_taskbot()
