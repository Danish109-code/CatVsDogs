import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Cat vs Dog AI Classifier",
    page_icon="🐱",
    layout="centered"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        color: #4A4A4A;
        font-size: 42px;
        font-weight: bold;
    }

    .subtitle {
        text-align: center;
        color: #777777;
        font-size: 18px;
    }

    .result-box {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        background-color: #f5f5f5;
        margin-top: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🐱 Cat vs Dog Classifier 🐶</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload an image and let the CNN model identify it.'
    '</div>',
    unsafe_allow_html=True
)

st.write("")


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_trained_model():

    model = load_model(
        "models/cat_dog_model.keras"
    )

    return model


try:

    model = load_trained_model()

except Exception as e:

    st.error(
        "❌ Could not load the trained model."
    )

    st.error(
        str(e)
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🤖 Model Information")

    st.write(
        "This application uses a "
        "**Convolutional Neural Network (CNN)** "
        "to classify images."
    )

    st.write("")

    st.write("**Input Size:** 224 × 224")

    st.write("**Classes:**")

    st.write("🐱 Cat")

    st.write("🐶 Dog")

    st.write("")

    st.info(
        "Upload a clear image of a cat or dog "
        "for better predictions."
    )


# ============================================================
# IMAGE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "📁 Upload an image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# PROCESS UPLOADED IMAGE
# ============================================================

if uploaded_file is not None:

    # --------------------------------------------------------
    # Open Image
    # --------------------------------------------------------

    image = Image.open(
        uploaded_file
    )

    # Convert to RGB
    image = image.convert(
        "RGB"
    )


    # --------------------------------------------------------
    # Display Image
    # --------------------------------------------------------

    st.subheader(
        "🖼️ Uploaded Image"
    )

    st.image(
        image,
        caption="Image to classify",
        width=450
    )


    st.write("")


    # ========================================================
    # PREDICTION BUTTON
    # ========================================================

    predict_button = st.button(
        "🔍 Predict Image",
        use_container_width=True
    )


    if predict_button:

        # ----------------------------------------------------
        # Show Processing Message
        # ----------------------------------------------------

        with st.spinner(
            "🤖 AI is analyzing the image..."
        ):

            # Resize
            processed_image = image.resize(
                (224, 224)
            )

            # Convert to NumPy
            image_array = np.array(
                processed_image
            )

            # Normalize
            image_array = (
                image_array / 255.0
            )

            # Add batch dimension
            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            # Make prediction
            prediction = model.predict(
                image_array,
                verbose=0
            )


        # ----------------------------------------------------
        # Get Probability
        # ----------------------------------------------------

        probability = float(
            prediction[0][0]
        )


        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        if probability >= 0.5:

            predicted_class = "Dog 🐶"

            confidence = (
                probability * 100
            )

            dog_probability = (
                probability * 100
            )

            cat_probability = (
                (1 - probability) * 100
            )

        else:

            predicted_class = "Cat 🐱"

            confidence = (
                (1 - probability) * 100
            )

            cat_probability = (
                (1 - probability) * 100
            )

            dog_probability = (
                probability * 100
            )


        # ====================================================
        # DISPLAY RESULT
        # ====================================================

        st.divider()

        st.subheader(
            "🎯 Prediction Result"
        )


        # ----------------------------------------------------
        # Main Prediction
        # ----------------------------------------------------

        if predicted_class == "Cat 🐱":

            st.success(
                f"Prediction: {predicted_class}"
            )

        else:

            st.info(
                f"Prediction: {predicted_class}"
            )


        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        st.write(
            f"### Confidence: {confidence:.2f}%"
        )

        st.progress(
            int(confidence)
        )


        # ====================================================
        # CLASS PROBABILITIES
        # ====================================================

        st.write("")

        st.subheader(
            "📊 Class Probabilities"
        )


        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "🐱 Cat",
                f"{cat_probability:.2f}%"
            )


        with col2:

            st.metric(
                "🐶 Dog",
                f"{dog_probability:.2f}%"
            )


        # ====================================================
        # LOW CONFIDENCE WARNING
        # ====================================================

        if confidence < 60:

            st.warning(
                "⚠️ The model is not very confident "
                "about this prediction. Try uploading "
                "a clearer image."
            )

        elif confidence >= 90:

            st.success(
                "✅ The model is highly confident "
                "about this prediction."
            )

        else:

            st.info(
                "ℹ️ The model has moderate confidence "
                "in this prediction."
            )


        # ====================================================
        # TECHNICAL INFORMATION
        # ====================================================

        with st.expander(
            "🔧 Show Technical Details"
        ):

            st.write(
                "**Raw Model Output:**"
            )

            st.write(
                prediction
            )

            st.write(
                "**Processed Image Shape:**"
            )

            st.write(
                image_array.shape
            )

            st.write(
                "**Model Interpretation:**"
            )

            st.write(
                "The model output represents the "
                "probability of the image belonging "
                "to the Dog class."
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Built with Python, TensorFlow, CNN and Streamlit 🚀"
)