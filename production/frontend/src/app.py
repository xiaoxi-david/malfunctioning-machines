import json
import os
import streamlit as st
import numpy as np

from PIL import Image
import utils
import requests

MAGE_EMOJI_URL = "https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/240/twitter/259/mage_1f9d9.png"

# Set page title and favicon.
st.set_page_config(
    page_title="Human vs computer",
    page_icon=MAGE_EMOJI_URL,
)

st.title("Human vs computer")
st.write(
    "Can you detect anomaly sounds in malfunctioning \
    industrial machines better than a computer?"
)

st.header("Step 1: Choose computer model")
model = st.selectbox(
    "Select model",
    options=["Supervised using spectogram", "Supervised using melspectogram"],
)

st.header("Step 2: Train yourself")
left, right = st.beta_columns(2)
with left:
    st.subheader("Test audio")
    test_fnames = utils.get_test_files()
    test_choices = list(range(len(test_fnames)))
    test_idx = st.select_slider(
        "Choose another audio", test_choices, key="test"
    )
    test_file = test_fnames[test_idx]
    label, machine_id, audio_id = utils.get_info(test_file)

    audio_file = open(test_file, "rb")
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format="audio/wav")
    with st.beta_expander("Test audio features"):
        test_img_path = utils.get_img(test_file)
        test_img = Image.open(test_img_path)
        st.image(test_img)

with right:
    st.subheader("Train audio")
    train_fnames = utils.get_train_files(machine_id)
    # st.write(train_fnames)
    train_choices = list(range(len(train_fnames)))
    train_idx = st.select_slider(
        "Choose another audio", train_choices, key="train"
    )
    train_file = train_fnames[train_idx]

    audio_file = open(train_file, "rb")
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format="audio/wav")
    with st.beta_expander("Train audio features"):
        train_img_path = utils.get_img(train_file)
        train_img = Image.open(train_img_path)
        st.image(train_img)

st.header("Step 3: Make your prediction")
guess = st.selectbox("Predict", ["Choose", "normal", "anomaly"])


st.header("Step 4: Check result")
with st.beta_expander("Result"):
    lab = st.empty()
    computer = st.empty()
    human = st.empty()
    winner = st.empty()

    if guess != "Choose":
        json_path = utils.get_json(train_file)

        with open(json_path) as f:
            data = json.load(f)
            json_data = json.dumps(data)

        headers = {"content-type": "application/json"}
        if model == "Supervised using spectogram":
            server = "http://tf-serving:8501/v1/models/stft:predict"
        elif model == "Supervised using melspectogram":
            server = "http://tf-serving:8501/v1/models/mel:predict"

        json_response = requests.post(
            server,
            data=json_data,
            headers=headers,
        )

        server_output = json.loads(json_response.text)["predictions"]
        prediction = ["normal", "anomaly"][server_output[0][0] > 0.5]

        lab.text(f"True value is {label}")
        computer.text(f"The model predicts {prediction}")
        human.text(f"You predicted {guess}")

        if guess == label and prediction == label:
            winner.text("Both win!")
            st.balloons()
        elif guess == label and prediction != label:
            winner.text("You win!")
            st.balloons()
        elif guess != label and prediction != label:
            winner.text("Both lose!")
        elif guess != label and prediction == label:
            winner.text("Computer wins!")
    else:
        winner.text("Choose")

with st.beta_expander("Explanation"):
    if guess != "Choose":
        if prediction == "normal":
            prob = 1 - server_output[0][0]
        elif prediction == "anomaly":
            prob = server_output[0][0]
        st.write(f"Model predicts {prediction} with a probability of {prob}")
    else:
        winner.text("Choose")