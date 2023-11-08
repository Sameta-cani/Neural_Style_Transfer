import streamlit as st
from PIL import Image
import style
import utils
import os

# 상수 정의
TEST_FOLDER = "neural_style"
IMAGE_CONTENT_FOLDER = f"{TEST_FOLDER}/images/content-images"
MODEL_FOLDER = f"{TEST_FOLDER}/saved_models"
IMAGE_OUTPUT_FOLDER = f"{TEST_FOLDER}/images/output-images"

# 기능별 함수화
def perform_style_transfer(input_image, output_image, model, style_name):
    model = style.load_model(model)
    data = style.stylize(model, input_image, output_image)
    data_img = data[0].clone().clamp(0, 255).numpy()
    data_img = data_img.transpose(1, 2, 0).astype("uint8")
    data_img = Image.fromarray(data_img)
    return data_img

st.title('PyTorch Style Transfer')

img = st.sidebar.selectbox(
    'Select Image',
    ('amber.jpg', 'cat.png', 'Image upload')
)

style_name = st.sidebar.selectbox(
    'Select Style',
    ('candy', 'mosaic', 'rain_princess', 'udnie')
)
st.write('### Source image:')

if img == 'Image upload':
    img_file = st.file_uploader('Upload an image.', type=['png', 'jpg', 'jpeg'])
    if img_file is not None:
        utils.save_uploaded_file(IMAGE_CONTENT_FOLDER, img_file)
        img = img_file.name

input_image = f"{IMAGE_CONTENT_FOLDER}/{img}"
output_image = f"{IMAGE_OUTPUT_FOLDER}/{style_name}-{img}"
model = f"{MODEL_FOLDER}/{style_name}.pth"

if os.path.exists(input_image):
    image = Image.open(input_image)
    st.image(image, width=400)

clicked = st.button('Stylize')

if clicked:
    with st.spinner(text="Waiting for Style Transfer"):
        styled_image = perform_style_transfer(input_image, output_image, model, style_name)
        st.write('### Output image:')
        st.image(styled_image, width=400)

        output_path = f"{style_name}-{img}"
        styled_image.save(output_path)
        with open(output_path, "rb") as file:
            st.download_button(
                label="Download image",
                data=file,
                file_name=output_path,
                mime="image/png"
            )
