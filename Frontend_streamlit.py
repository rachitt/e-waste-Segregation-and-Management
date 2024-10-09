import glob
import streamlit as st
import wget
from PIL import Image
import torch
import cv2
import os
import time
import numpy as np
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

st.set_page_config(layout="wide")

cfg_model_path = 'models/yolov5s.pt'
model = None
confidence = .25


def image_input(data_src):
    img_file = None
    if data_src == 'Sample data':
        img_path = glob.glob('data/sample_images/*')
        img_slider = st.slider("Select a test image.", min_value=1, max_value=len(img_path), step=1)
    else:
        img_bytes = st.sidebar.file_uploader("Upload an image", type=['png', 'jpeg', 'jpg'])
        if img_bytes:
            img_file = "data/uploaded_data/upload." + img_bytes.name.split('.')[-1]
            Image.open(img_bytes).save(img_file)

    if img_file:
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_file, caption="Selected Image")
        with col2:
            img = infer_image(img_file)
            st.image(img, caption="Model prediction")
        st.header('Count of Objects:')
        counts = count_objects(model(img))
        for class_name, count in counts.items():
            st.subheader(f"{class_name}: {count}")
            if class_name == "damaged-bulb" or class_name == "damaged-battery":
                color = 'blue'

                st.write(f'<h3 style="color:{color}">segregate {count} {class_name} from waste pile</h3>',
                         unsafe_allow_html=True)


def video_input(data_src):
    vid_file = None
    if data_src == 'Sample data':
        vid_file = "data/sample_videos/sample.mp4"
    else:
        vid_bytes = st.sidebar.file_uploader("Upload a video", type=['mp4', 'mpv', 'avi'])
        if vid_bytes:
            vid_file = "data/uploaded_data/upload." + vid_bytes.name.split('.')[-1]
            with open(vid_file, 'wb') as out:
                out.write(vid_bytes.read())

    if vid_file:
        cap = cv2.VideoCapture(vid_file)
        custom_size = st.sidebar.checkbox("Custom frame size")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if custom_size:
            width = st.sidebar.number_input("Width", min_value=120, step=20, value=width)
            height = st.sidebar.number_input("Height", min_value=120, step=20, value=height)

        fps = 0
        st1, st2, st3 = st.columns(3)
        with st1:
            st.markdown("## Height")
            st1_text = st.markdown(f"{height}")
        with st2:
            st.markdown("## Width")
            st2_text = st.markdown(f"{width}")
        with st3:
            st.markdown("## FPS")
            st3_text = st.markdown(f"{fps:.2f}")

        st.markdown("---")
        output = st.empty()
        
        prev_time = 0
        curr_time = 0
        unique_objects = {}
        counted_objects = set()  # Set to store IDs of counted objects
        frame_count = 0
        # Process frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            frame = cv2.resize(frame, (width, height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output_img = infer_image(frame)
            output.image(output_img)
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            st1_text.markdown(f"**{height}**")
            st2_text.markdown(f"**{width}**")
            st3_text.markdown(f"**{fps:.2f}**")
            detections = model(frame)
            for det in detections.xyxy[0]:
                class_id = int(det[5])
                if class_id not in counted_objects:  # Check if object has already been counted
                    class_name = model.names[class_id]
                    # Increment count for the class
                    if class_name in unique_objects:
                        unique_objects[class_name] += 1
                    else:
                        unique_objects[class_name] = 1
                    # Add object ID to counted set
                    counted_objects.add(class_id)

            

            

        cap.release()

        st.header('Count of Objects:')
        for class_name, count in unique_objects.items():
            st.subheader(f"{class_name}: {count}")
            if class_name == "damaged-bulb" or class_name == "damaged-battery":
                color = 'blue'

                st.write(f'<h3 style="color:{color}">segregate {count} {class_name} from waste pile</h3>',
                         unsafe_allow_html=True)


def infer_image(img, size=None):
    model.conf = confidence
    result = model(img, size=size) if size else model(img)
    result.render()
    image = Image.fromarray(result.ims[0])
    return image


@st.cache_resource
def load_model(path, device):
    model_ = torch.hub.load('ultralytics/yolov5', 'custom', path=path, force_reload=True)
    model_.to(device)
    print("model to ", device)
    return model_


@st.cache_resource
def download_model(url):
    model_file = wget.download(url, out="models")
    return model_file


def get_user_model():
    model_src = st.sidebar.radio("Model source", ["file upload", "url"])
    model_file = None
    if model_src == "file upload":
        model_bytes = st.sidebar.file_uploader("Upload a model file", type=['pt'])
        if model_bytes:
            model_file = "models/uploaded_" + model_bytes.name
            with open(model_file, 'wb') as out:
                out.write(model_bytes.read())
    else:
        url = st.sidebar.text_input("model url")
        if url:
            model_file_ = download_model(url)
            if model_file_.split(".")[-1] == "pt":
                model_file = model_file_

    return model_file


def count_objects(detections):
    counts = {}
    for det in detections.xyxy[0]:
        class_id = int(det[5])
        class_name = model.names[class_id]
        if class_name in counts:
            counts[class_name] += 1
        else:
            counts[class_name] = 1
    return counts


def main():
    global model, confidence, cfg_model_path

    st.title("Waste Recognition Dashboard")

    st.sidebar.title("Settings")

    model_src = st.sidebar.radio("Select yolov5 weight file", ["Use demo model 5s", "Use your own model"])
    if model_src == "Use your own model":
        user_model_path = get_user_model()

        if user_model_path:
            cfg_model_path = user_model_path

        st.sidebar.text(cfg_model_path.split("/")[-1])
        st.sidebar.markdown("---")

    if not os.path.isfile(cfg_model_path):
        st.warning("Model file not available!!!, please added to the model folder.")
    else:
        if torch.cuda.is_available():
            device_option = st.sidebar.radio("Select Device", ['cpu', 'cuda'], disabled=False, index=0)
        else:
            device_option = st.sidebar.radio("Select Device", ['cpu', 'cuda'], disabled=True, index=0)

        model = load_model(cfg_model_path, device_option)

        confidence = st.sidebar.slider('Confidence', min_value=0.1, max_value=1.0, value=.45)

        if st.sidebar.checkbox("Custom Classes"):
            model_names = list(model.names.values())
            assigned_class = st.sidebar.multiselect("Select Classes", model_names, default=[model_names[0]])
            classes = [model_names.index(name) for name in assigned_class]
            model.classes = classes
        else:
            model.classes = list(model.names.keys())

        st.sidebar.markdown("---")

        input_option = st.sidebar.radio("Select input type: ", ['image', 'video'])

        data_src = st.sidebar.radio("Select input source: ", ['Sample data', 'Upload your own data'])

        if input_option == 'image':
            image_input(data_src)
        else:
            video_input(data_src)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
