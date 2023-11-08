import torch
from PIL import Image
import os
import streamlit as st
import style

# 기능별 함수화
def perform_style_transfer(input_image, output_image, model, style_name):
    model = style.load_model(model)
    data = style.stylize(model, input_image, output_image)
    data_img = data[0].clone().clamp(0, 255).numpy()
    data_img = data_img.transpose(1, 2, 0).astype("uint8")
    data_img = Image.fromarray(data_img)
    return data_img

def load_image(filename, size=None, scale=None):
    img = Image.open(filename)
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img


def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std

# 파일 업로드 함수
def save_uploaded_file(directory, file):
	if not os.path.exists(directory):
		os.makedirs(directory)
	with open(os.path.join(directory, file.name), 'wb') as f:
		f.write(file.getbuffer())
	return st.success('파일 업로드 성공')
