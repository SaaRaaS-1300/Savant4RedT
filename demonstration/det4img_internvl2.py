import base64
import io
from dataclasses import asdict, dataclass
from typing import List, Optional

import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers.utils import logging
# from ipex_llm.transformers import AutoModel
from transformers import AutoModel
from transformers import AutoTokenizer  # isort: skip


# 添加自定义 CSS 样式
st.markdown(
    """
    <style>
    .stChatInput {
        background-color: #e6e6fa; /* 浅淡紫色 */
    }
    .stDownloadButton {
        background-color: #f5f5f5;
        color: black;
        border: 0px solid #000;
        border-radius: 5px;
        text-align: center;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
        cursor: pointer;
        transition: background-color 1.2s;
        display: block;
        margin: 20px auto;
        text-align: center;
    }
    .stDownloadButton:hover {
        background-color: #ff9999;
    }
    </style>
    """,
    unsafe_allow_html=True
)

MODEL_PATH = 'models/InternVL2-2B-finetuned'
USER_AVATAR = 'resource/demo_pic/pic_user.png'
ROBOT_AVATAR = 'resource/demo_pic/pic_bot.png'
logger = logging.get_logger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


@dataclass
class GenerationConfig:
    # this config is used for chat to provide more diversity
    max_length: int = 32768
    top_p: float = 0.8
    temperature: float = 0.8
    do_sample: bool = True
    repetition_penalty: float = 1.005


@torch.inference_mode()
def generate_interactive(
    model,
    tokenizer,
    prompt,
    history: List[str],
    image: Optional[str] = None,
    generation_config: Optional[GenerationConfig] = None,
):
    if generation_config is None:
        generation_config = GenerationConfig()

    # prepare the image
    if image is not None:
        pixel_values = load_image(image,
                                  max_num=12).to(torch.bfloat16).cuda()
    else:
        pixel_values = None
    
    # generate
    result = model.chat(tokenizer=tokenizer,
                        pixel_values=pixel_values,
                        question=prompt,
                        history=history,
                        return_history=False,
                        generation_config=asdict(generation_config))

    # output
    for i in range(len(result)):
        yield result[:i + 1]


def on_btn_click():
    del st.session_state.internvl2_messages
    del st.session_state.internvl2_image


@st.cache_resource
def load_model():
    model = (AutoModel.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True, load_in_4bit=False).to(torch.bfloat16).cuda())
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH,
                                              trust_remote_code=True)
    return model, tokenizer


def prepare_generation_config():
    with st.sidebar:
        max_length = st.slider('Max Length',
                               min_value=8,
                               max_value=16384,
                               value=8192)
        top_p = st.slider('Top P', 0.0, 1.0, 0.6, step=0.01)
        temperature = st.slider('Temperature', 0.0, 1.0, 0.6, step=0.01)
        st.file_uploader('Upload Image', type=['jpg', 'jpeg', 'png'], key='internvl2_image')
        st.button('Clear Chat History', on_click=on_btn_click)

    generation_config = GenerationConfig(max_length=max_length,
                                         top_p=top_p,
                                         temperature=temperature)

    return generation_config


def modify_system_message(model):
    model.system_message = '你是 AI 信息内容安全专家'


def generate_markdown():
    messages = st.session_state.internvl2_messages
    image = st.session_state.internvl2_image
    markdown_content = ""
    for i, message in enumerate(messages):
        role = message['role']
        content = message['content']
        if role == 'user':
            markdown_content += f"**User:**\n\n{content}\n\n"
            if i == 0 and image:
                image = Image.open(image).convert('RGB')
                width, height = image.size
                ratio = 256 / max(width, height)
                image = image.resize((int(width * ratio), int(height * ratio)))
                image_data = io.BytesIO()
                image.save(image_data, format='PNG')
                base64_string = base64.b64encode(image_data.getvalue()).decode('utf-8')
                markdown_content += f'<img src="data:image/png;base64,{base64_string}"/>\n\n'
        elif role == 'robot':
            markdown_content += f"**Robot:**\n\n{content}\n\n"
            if '安全等级划分' in content:
                # 安全等级划分：[低]危险等级
                level = content.split('安全等级划分：[')[1].split(']')[0]
                if level == '高':
                    markdown_content = markdown_content.replace('<img', '<!--\n<img')
                    markdown_content = markdown_content.replace('/>', '/>\n-->')
    return markdown_content


def main():
    print('load models begin.')
    model, tokenizer = load_model()
    modify_system_message(model)
    print('load models end.')
    
    st.title('🛡️ Savant4RedT && 内容安全测试')
    st.sidebar.markdown('## 🍏 Model Configuration')
    
    generation_config = prepare_generation_config()
    
    # Initialize chat history
    if 'internvl2_messages' not in st.session_state:
        st.session_state.internvl2_messages = []
    if 'internvl2_image' not in st.session_state:
        st.session_state.internvl2_image = None
    
    # Display image from history on app rerun
    if st.session_state.internvl2_image:
        st.image(st.session_state.internvl2_image)
    
    # Display chat messages from history on app rerun
    for message in st.session_state.internvl2_messages:
        with st.chat_message(message['role'], avatar=message.get('avatar')):
            st.markdown(message['content'])
    
    # Accept user input
    if prompt := st.chat_input('本生成式模型接口由 Savant4RedT-1.8B-Content 进行推理'):
        # Display user message in chat message container
        with st.chat_message('user', avatar=USER_AVATAR):
            prompt = f'请你分析以上图片：\n<image>'
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.internvl2_messages.append({
            'role': 'user',
            'content': prompt,
            'avatar': USER_AVATAR
        })
    
        with st.chat_message('robot', avatar=ROBOT_AVATAR):
            message_placeholder = st.empty()
            for cur_response in generate_interactive(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    history=st.session_state.internvl2_messages[:-1],
                    image=st.session_state.internvl2_image,
                    generation_config=generation_config,
            ):
                # Display robot response in chat message container
                message_placeholder.markdown(cur_response + '▌')
            message_placeholder.markdown(cur_response)
        # Add robot response to chat history
        st.session_state.internvl2_messages.append({
            'role': 'robot',
            'content': cur_response,  # pylint: disable=undefined-loop-variable
            'avatar': ROBOT_AVATAR,
        })
    
    # Generate and provide download link for markdown file
    if st.session_state.internvl2_messages:
        markdown_content = generate_markdown()
        st.download_button(
            label="🐬 Download the Analyzing Result as Markdown Files 🐬",
            data=markdown_content,
            file_name="analyzing_result.md",
            mime="text/markdown"
        )


main()
