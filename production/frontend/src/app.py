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
left, middle, right = st.beta_columns(3)
with left:
    feature = st.selectbox("Select feature", options=["spectogram", "melspectogram"])
with middle:
    architecture = st.selectbox("Select architecture", options=["AE", "CAE"])
with right:
    function = st.selectbox("Select function", options=["fun1", "fun2"])

st.header("Step 2: Train yourself")
left, right = st.beta_columns(2)
with left:
    st.subheader("Test audio")
    test_fnames = utils.get_files("test")
    test_choices = list(range(len(test_fnames)))
    test_idx = st.select_slider("Choose another audio", test_choices, key="test")
    test_file = test_fnames[test_idx]
    label, machine_id, audio_id = utils.get_info(test_file)
    st.write(test_file)  # !For debugging
    st.write(label)  # !For debugging

    audio_file = open(test_file, "rb")
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format="audio/wav")
    with st.beta_expander("Test audio features"):
        test_img_dir = utils.get_feature_img(test_file)
        test_img = Image.open(test_img_dir)
        st.write(test_img_dir)  # !For debugging
        st.image(test_img)

with right:
    st.subheader("Train audio")
    train_fnames = utils.get_train_files(machine_id)
    train_choices = list(range(len(train_fnames)))
    train_idx = st.select_slider("Choose another audio", train_choices, key="train")
    train_file = test_fnames[train_idx]
    st.write(train_file)  # !For debugging
    st.write(label)  # !For debugging

    audio_file = open(train_file, "rb")
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format="audio/wav")
    with st.beta_expander("Train audio features"):
        train_img_dir = utils.get_feature_img(train_file)
        train_img = Image.open(train_img_dir)
        st.write(train_img_dir)  # !For debugging
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
        prediction = np.random.choice(
            ["anomaly", "normal"]
        )  # TODO: Replace it with actual model prediction

        lab.text(f"True value is {label}")
        computer.text(
            f"The model {architecture} with {feature} and {function} predicts {prediction}"
        )
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
    st.write("Reconstruction error was: ")
    # TODO: Extend with actual model prediction:
    # TODO: reconstruction error, comparison within the class, plots
    dir_root = os.path.join("..", "media", "json")

    fname = "anomaly_id_04_00000080.json"
    label, machine_id, audio_id = utils.get_info(fname)

    st.write(f"True label: {label}")
    with open(os.path.join(dir_root, fname)) as f:
        data = json.load(f)
        json_data = json.dumps(data)

    headers = {"content-type": "application/json"}
    json_response = requests.post(
        "http://tf-serving:8501/v1/models/mel:predict",
        data=json_data,
        headers=headers,
    )

    predictions = json.loads(json_response.text)["predictions"]
    pred_label = ["normal", "anomaly"][predictions[0][0] > 0.5]
    st.write(f"Machine predicts: {pred_label}")

    fname = "normal_id_04_00000025.json"
    label, machine_id, audio_id = utils.get_info(fname)

    st.write(f"True label: {label}")
    with open(os.path.join(dir_root, fname)) as f:
        data = json.load(f)
        json_data = json.dumps(data)

    headers = {"content-type": "application/json"}
    json_response = requests.post(
        "http://tf-serving:8501/v1/models/mel:predict",
        data=json_data,
        headers=headers,
    )

    predictions = json.loads(json_response.text)["predictions"]
    pred_label = ["normal", "anomaly"][predictions[0][0] > 0.5]
    st.write(f"Machine predicts: {pred_label}")
