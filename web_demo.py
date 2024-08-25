import copy
import warnings
from dataclasses import asdict, dataclass
from typing import Callable, List, Optional
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import streamlit as st
import torch
from torch import nn
from transformers.generation.utils import (LogitsProcessorList,
                                           StoppingCriteriaList)
from transformers.utils import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModel, AutoTokenizer
from PIL import Image
from transformers import TextIteratorStreamer
from threading import Thread

logger = logging.get_logger(__name__)

model_name_or_path = "/root/Project/ReceipeLLM/InternVL2-2B-Receipe"
logo = Image.open("logo.png")

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

def load_upload_file_and_show(uploaded_file):
    pixel_values = None
    if uploaded_file is not None:
        pixel_values = load_image(uploaded_file, max_num=12).to(torch.bfloat16).cuda()
    return pixel_values

@dataclass
class GenerationConfig:
    max_length: int = 2048
    top_p: float = 0.75
    temperature: float = 0.1
    do_sample: bool = True
    repetition_penalty: float = 1.000

@torch.inference_mode()
def generate_interactive(
    model,
    tokenizer,
    prompt,
    pixel_values,
    **kwargs,
):
    # 初始化
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=10)
    generation_config = dict(max_new_tokens=1024, do_sample=False, streamer=streamer)
    thread = Thread(target=model.chat, kwargs=dict(
        tokenizer=tokenizer, pixel_values=pixel_values, question=prompt,
        history=None, return_history=False, generation_config=generation_config, ))
    thread.start()
    generated_text = ''
    for new_text in streamer:
        if new_text == model.conv_template.sep:
            break
        generated_text += new_text
        yield generated_text

def on_btn_click():
    del st.session_state.messages

@st.cache_resource
def load_model():
    model = AutoModel.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True,
                                              use_fast=False)
    return model, tokenizer

user_prompt = 'user\n{user}\n'
robot_prompt = 'assistant\n{robot}\n'
cur_query_prompt = 'user\n{user}\n    assistant\n'

def combine_history(prompt):
    messages = st.session_state.messages
    meta_instruction = ('')
    total_prompt = f"<s>system\n{meta_instruction}\n"
    for message in messages:
        cur_content = message['content']
        if message['role'] == 'user':
            cur_prompt = user_prompt.format(user=cur_content)
        elif message['role'] == 'robot':
            cur_prompt = robot_prompt.format(robot=cur_content)
        else:
            raise RuntimeError
        total_prompt += cur_prompt
    total_prompt = total_prompt + cur_query_prompt.format(user=prompt)
    return total_prompt

def clear_file_uploader():
    st.session_state.uploader_key += 1
    # st.rerun()

def main():
    global pixel_values
    print('load model begin.')
    model, tokenizer = load_model()
    print('load model end.')

    if 'uploader_key' not in st.session_state:
        st.session_state.uploader_key = 0
    
    # 侧边栏
    with st.sidebar:
        st.image(logo, caption='', use_column_width=True)
        lan = st.selectbox('#### Language / 语言', ['English', '中文'], on_change=st.rerun,
                       help='This is only for switching the UI language. 这仅用于切换UI界面的语言。')
        if lan == 'English':
            with st.expander('🔥 Advanced Options'):
                temperature = st.slider('temperature', min_value=0.0, max_value=1.0, value=0.7, step=0.1)
                top_p = st.slider('top_p', min_value=0.0, max_value=1.0, value=0.95, step=0.05)
                repetition_penalty = st.slider('repetition_penalty', min_value=1.0, max_value=1.5, value=1.1, step=0.02)
                max_length = st.slider('max_new_token', min_value=0, max_value=4096, value=1024, step=128)
                max_input_tiles = st.slider('max_input_tiles (control image resolution)', min_value=1, max_value=24,
                                            value=12, step=1)
            st.button('Clear History', on_click=on_btn_click)

            uploaded_image = st.file_uploader('Upload a file',
                                          type=['png', 'jpg', 'jpeg', 'webp'],
                                          help='You can upload an image',
                                          key=f'uploader_{st.session_state.uploader_key}',
                                          on_change=st.rerun)
            todo_list = st.sidebar.selectbox('Our to-do list', ['👏This is our to-do list',
                                        '1. More speed',
                                        '2. Speech generation',
                                        '3. More data for finetuning'], key='todo_list',
                        help='Here are some features we plan to support in the future.')
        else:
            with st.expander('🔥 高级选项'):
                temperature = st.slider('temperature', min_value=0.0, max_value=1.0, value=0.7, step=0.1)
                top_p = st.slider('top_p', min_value=0.0, max_value=1.0, value=0.95, step=0.05)
                repetition_penalty = st.slider('重复惩罚', min_value=1.0, max_value=1.5, value=1.1, step=0.02)
                max_length = st.slider('最大输出长度', min_value=0, max_value=4096, value=1024, step=128)
                max_input_tiles = st.slider('最大图像块数 (控制图像分辨率)', min_value=1, max_value=24, value=12, step=1)
            st.button('清空聊天历史', on_click=on_btn_click)

            # 更新uploader_key有助于不重复输出图片
            uploaded_image = st.file_uploader('上传一张图片',
                                type=['png', 'jpg', 'jpeg', 'webp'],
                                help='你可以上传一张图片',
                                key=f'uploader_{st.session_state.uploader_key}',
                                on_change=st.rerun)

            todo_list = st.sidebar.selectbox('我们的待办事项', ['👏这里是我们的待办事项', '1. 更快推理速度',
                                '2. 支持语音输出', '3. 更多数据用于微调'], key='todo_list',
                    help='这是我们计划要支持的一些功能。')

        # Image uploader in the sidebar
        
        pixel_values = load_upload_file_and_show(uploaded_image)
        if pixel_values is not None:
            st.session_state.uploaded_image = uploaded_image
    
    if lan == "English":
        st.title("🍲Chinese Receipe Generation🍲")
        sys_prompt = "Hello, I am the Chinese Cuisine Recipe Model 🍲. You can upload an image 🍳, and I will analyze how it was made."
    else:
        st.title('🍲中华食谱大模型🍲')
        sys_prompt = "您好，我是中华食谱大模型🍲，您可以上传一张图片🍳，我会为您分析它是如何制作的。"
    
    # if 'messages' not in st.session_state:
    #         st.session_state.messages = [{
    #             'role': 'robot',
    #             'content': sys_prompt
    #         }]
    # else:
    if 'messages' not in st.session_state:
        st.session_state.messages = [{
                'role': 'robot',
                'content': sys_prompt
        }]
    else:
        st.session_state.messages[0]["content"] = sys_prompt

    generation_config = GenerationConfig(max_length=max_length,
                                         top_p=top_p,
                                         temperature=temperature)

    for message in st.session_state.messages:
        with st.chat_message(message['role'], avatar=message.get('avatar')):
            st.markdown(message['content'])
            if "image" in message.keys():
                st.image(message['image'], caption='', use_column_width=True)

    
    # 监听用户输入
    if lan == "English":
        prompt = st.chat_input('What is up?')
    else:
        prompt = st.chat_input('请上传食物图片，输入你想问的问题...')
    if prompt:
        with st.chat_message('user'):
            # 输出内容
            st.markdown(prompt)
        
        # real_prompt = combine_history(prompt)

        user_message = {
            'role': 'user',
            'content': prompt
        }

        if pixel_values != None:
            # st.text("add image")
            user_message['image'] = st.session_state.uploaded_image
            st.image(user_message['image'], caption='', use_column_width=True)
            clear_file_uploader()
        else:
            # 没有上传图片，从之前的信息中找
            for message in st.session_state.messages:
                if "image" in message.keys():
                    image = Image.open(message['image'])
                    # 获取图片的宽度和高度
                    pixel_values = load_image(message['image'], max_num=12).to(torch.bfloat16).cuda()
                    break
        

        st.session_state.messages.append(user_message)

        with st.chat_message('robot'):
            message_placeholder = st.empty()
            for cur_response in generate_interactive(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    pixel_values = pixel_values,
                    **asdict(generation_config),
            ):
                message_placeholder.markdown(cur_response + '▌')
            message_placeholder.markdown(cur_response)
        
        st.session_state.messages.append({
            'role': 'robot',
            'content': cur_response,
        })

if __name__ == '__main__':
    main()

