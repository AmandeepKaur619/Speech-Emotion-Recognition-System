import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
import plotly.graph_objects as go
import plotly.express as px

# -------------------------------------
# Load Model, Scaler, Label Encoder
# -------------------------------------
model = tf.keras.models.load_model("ser_model.keras")
label_encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")

# -------------------------------------
# Emotion Colors
# -------------------------------------
EMOTION_COLORS = {
    "angry": "#FF4C4C",
    "sad": "#4C6BFF",
    "fearful": "#6A0DAD",
    "happy": "#FFD93D",
    "surprised": "#32CD32",
    "calm": "#00CED1",
    "neutral": "#A9A9A9",
    "disgust": "#8B0000",
}

# -------------------------------------
# Feature Extractor
# -------------------------------------
def extract_features(file_path, max_pad_len=174):
    try:
        audio, sample_rate = librosa.load(file_path, res_type="kaiser_fast")
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20)

        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode="constant")
        else:
            mfcc = mfcc[:, :max_pad_len]

        return mfcc.flatten(), audio, sample_rate

    except Exception as e:
        print("Error:", e)
        return None, None, None


# -------------------------------------
# Streamlit Page Config
# -------------------------------------
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="",
    layout="wide"
)

# -------------------------------------
# Sidebar
# -------------------------------------
st.sidebar.title(" About SER App")
st.sidebar.info("upload your .wav file"
    
)

st.sidebar.markdown("### Available Emotions:")
for emo, col in EMOTION_COLORS.items():
    st.sidebar.markdown(f"- <span style='color:{col}'>{emo}</span>", unsafe_allow_html=True)




# -------------------------------------
# Header
# -------------------------------------
st.markdown(
    """
    <div style="
        background: linear-gradient(90deg, #4b6cb7, #182848);
        padding: 25px;
        border-radius: 12px;
        text-align:center;
        color:white;
        font-size:32px;
        font-weight:bold;
    ">
         Speech Emotion Recognition (SER)
    </div>
    """,
    unsafe_allow_html=True
)

st.write("### Upload an audio file and visualize its emotional characteristics!")

uploaded_file = st.file_uploader(" Upload audio (.wav)", type=["wav"])

if uploaded_file:

    # Save temporarily
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    features, audio, sr = extract_features("temp.wav")

    if features is None:
        st.error(" Error extracting features.")
        st.stop()

    features_scaled = scaler.transform([features])

    with st.spinner(" Analyzing emotion..."):
        prediction = model.predict(features_scaled)

    predicted_idx = np.argmax(prediction)
    emotion = label_encoder.inverse_transform([predicted_idx])[0]

    # -------------------------------------
    # Main Detected Emotion Box
    # -------------------------------------
    st.markdown(
        f"""
        <div style="
            background-color:{EMOTION_COLORS.get(emotion)};
            padding:25px;
            border-radius:15px;
            text-align:center;
            color:white;
            font-size:30px;
            font-weight:bold;
        ">
             Detected Emotion: {emotion.upper()}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.audio(uploaded_file)

    # -------------------------------------
    # Probability Bar Chart
    # -------------------------------------
    st.subheader(" Emotion Probability Distribution")

    emotions = label_encoder.classes_
    probs = prediction[0]

    fig_bar = px.bar(
        x=emotions,
        y=probs,
        color=emotions,
        color_discrete_map=EMOTION_COLORS,
        labels={"x": "Emotion", "y": "Probability"},
        title="Model Confidence",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # -------------------------------------
    # Radar Chart
    # -------------------------------------
    st.subheader("Radar Visualization")
    fig_radar = go.Figure()

    fig_radar.add_trace(go.Scatterpolar(
        r=probs,
        theta=emotions,
        fill='toself',
        line=dict(color="gold", width=2)
    ))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=False,
    )

    st.plotly_chart(fig_radar, use_container_width=True)

    # -------------------------------------
    # Waveform
    # -------------------------------------
    st.subheader(" Waveform")
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(audio, sr=sr, ax=ax)
    ax.set_title("Waveform")
    st.pyplot(fig)

    # -------------------------------------
    # Spectrogram
    # -------------------------------------
    st.subheader(" Spectrogram")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    stft = librosa.stft(audio)
    spec = librosa.amplitude_to_db(abs(stft))
    img = librosa.display.specshow(spec, sr=sr, x_axis="time", y_axis="log", ax=ax2)
    fig2.colorbar(img, ax=ax2)
    ax2.set_title("Spectrogram (dB)")
    st.pyplot(fig2)

    # -------------------------------------
    # MFCC Heatmap
    # -------------------------------------
    st.subheader(" MFCC Heatmap")
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    img2 = librosa.display.specshow(mfcc, x_axis="time", ax=ax3)
    fig3.colorbar(img2, ax=ax3)
    ax3.set_title("MFCC (20 Coefficients)")
    st.pyplot(fig3)

