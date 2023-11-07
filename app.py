import streamlit as st
import openai
import func
import diffusers
from diffusers import utils
import torch
import cv2
from PIL import Image
import numpy as np
import os

# 파일 업로드 함수
def save_uploaded_file(directory, file):
	if not os.path.exists(directory):
		os.makedirs(directory)
	with open(os.path.join(directory, file.name), 'wb') as f:
		f.write(file.getbuffer())
	return st.success('파일 업로드 성공')

openai.api_key = st.secrets["api_key"]
	
controlnet_model = "lllyasviel/sd-controlnet-canny"
sd_model = "Lykon/DreamShaper"

controlnet = diffusers.ControlNetModel.from_pretrained(
    controlnet_model,
    torch_dtype=torch.float16
)
pipe = diffusers.StableDiffusionControlNetPipeline.from_pretrained(
    sd_model,
    controlnet=controlnet,
    torch_dtype=torch.float16
)

pipe.scheduler = diffusers.PNDMScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

st.write(pipe)

def img2img(img_path, prompt, negative_prompt, num_steps=20, guidance_scale=7, seed=0, low=100, high=200):
    image = diffusers.utils.load_image(img_path)

    np_image = np.array(image)

    canny_image = cv2.Canny(np_image, low, high)

    canny_image = canny_image[:, :, None]
    canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
    canny_image = Image.fromarray(canny_image)

    out_image = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=torch.manual_seed(seed),
        image=canny_image
    ).images[0]

    return image, canny_image, out_image

prompt = "masterpiece, best quality, ultra-detailed, illustration, school uniform, scarf, gymnasium"
negative_prompt = "lowres, ((bad anatomy)), ((bad hands)), text, missing finger, extra digits, fewer digits, blurry, ((mutated hands and fingers)), (poorly drawn face), ((mutation)), ((deformed face)), (ugly), ((bad proportions)), ((extra limbs)), extra face, (double head), (extra head), ((extra feet)), monster, logo, cropped, worst quality, low quality, normal quality, jpeg, humpbacked, long body, long neck, ((jpeg artifacts))"
num_steps = 20
guidance_scale = 7
seed = 3467120481370323442

st.title("ChatGPT Plus DALL-E")

img_file = st.file_uploader('이미지를 업로드 하세요.', type=['png', 'jpg', 'jpeg'])
if img_file is not None:
    st.write(type(img_file))
    st.write(img_file.name)
    st.write(img_file.size)
    st.write(img_file.type)

    save_uploaded_file('image', img_file)
	
    st.image(f'image/{img_file.name}')

    image, canny_image, out_image = img2img(f'image/{img_file.name}', prompt, negative_prompt, num_steps, guidance_scale, seed)
	
    st.image(out_image)

with st.form(key="form"):
    user_input = st.text_input(label="Prompt")
    size = st.selectbox("Size", ["1024x1024", "512x512", "256x256"])
    submit = st.form_submit_button(label="Submit")

if submit and user_input:
    gpt_prompt = [{
        "role": "system",
        "content": "Imagine the detail appeareance of the input. Response it shortly around 20 words."
    }]

    gpt_prompt.append({
        "role": "user",
        "content": user_input
    })
    with st.spinner(text="Waiting for ChatGPT..."):
        gpt_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=gpt_prompt
        )

    prompt = gpt_response["choices"][0]["message"]["content"]
    st.write(prompt)

    with st.spinner(text="Waiting for DALL-E..."):
        dalle_response = openai.Image.create(
            prompt=prompt,
            size=size
        )
    
    st.image(dalle_response["data"][0]["url"])