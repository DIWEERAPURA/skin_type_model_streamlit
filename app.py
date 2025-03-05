import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="DermAI • Skin Intelligence Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items=None
)

# Advanced CSS with Material Design influences
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Google+Sans:ital,wght@0,400;0,500;0,700;1,400&display=swap');

* {{
    font-family: 'Google Sans', sans-serif;
    box-sizing: border-box;
}}

:root {{
    --primary: #1A73E8;
    --surface: #FFFFFF;
    --background: #F8F9FA;
    --on-surface: #202124;
    --secondary-text: #5F6368;
}}

[data-theme="dark"] {{
    --surface: #2D2F31;
    --background: #202124;
    --on-surface: #E8EAED;
    --secondary-text: #9AA0A6;
}}

body {{
    background: var(--background);
    color: var(--on-surface);
}}

.stApp {{
    background-image: radial-gradient(circle at 1px 1px, var(--secondary-text) 1px, transparent 0);
    background-size: 40px 40px;
}}

.main-container {{
    max-width: 1280px;
    margin: 4rem auto;
    padding: 0 2rem;
}}

.upload-card {{
    background: var(--surface);
    border: 1px solid #DFE1E5;
    border-radius: 24px;
    padding: 4rem;
    text-align: center;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 1px 2px rgba(0,0,0,.08);
}}

.upload-card:hover {{
    border-color: var(--primary);
    box-shadow: 0 8px 24px rgba(66,133,244,.15);
}}

.result-card {{
    background: var(--surface);
    border-radius: 20px;
    padding: 2rem;
    margin-top: 2rem;
    box-shadow: 0 3px 6px rgba(0,0,0,.05);
}}

.confidence-meter {{
    height: 8px;
    background: rgba(26,115,232,.12);
    border-radius: 8px;
    overflow: hidden;
    position: relative;
}}

.confidence-fill {{
    height: 100%;
    background: var(--primary);
    width: 0;
    transition: width 1s ease;
    border-radius: 8px;
}}

.prediction-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 1.5rem;
    margin-top: 2rem;
}}

.prediction-card {{
    padding: 1.5rem;
    border-radius: 16px;
    background: rgba(26,115,232,.04);
    transition: all 0.2s ease;
}}

.prediction-card.active {{
    background: rgba(26,115,232,.12);
    border: 2px solid var(--primary);
}}

.analysis-button {{
    background: var(--primary) !important;
    color: white !important;
    border: none !important;
    padding: 14px 24px !important;
    border-radius: 20px !important;
    font-weight: 500 !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
}}

.analysis-button:hover {{
    opacity: 0.9;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(26,115,232,.3);
}}

.spinner {{
    border: 3px solid rgba(26,115,232,.1);
    border-top-color: var(--primary);
    animation: spin 1s linear infinite;
}}

@keyframes spin {{
    to {{ transform: rotate(360deg); }}
}}
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown("""
    <div class="main-container">
        <div style="text-align: center; margin-bottom: 4rem;">
            <h1 style="font-size: 3.5rem; margin-bottom: 1rem; color: var(--on-surface);">
                Advanced Skin Analysis
            </h1>
            <p style="color: var(--secondary-text); font-size: 1.1rem; max-width: 680px; margin: 0 auto;">
                Dermatological intelligence powered by deep learning. Analyze skin conditions with clinical-grade accuracy.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Model loader
    @st.cache_resource
    def load_model():
        try:
            return tf.keras.models.load_model('skin_type_model_final.h5')
        except Exception as e:
            st.error(f"Model initialization failed: {str(e)}")
            return None

    model = load_model()

    # Image processor
    def preprocess_image(image):
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        return np.expand_dims(img_array, axis=0)

    # Upload interface
    with st.container():
        st.markdown('<div class="upload-card">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drag dermatological image here or browse files",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # Analysis workflow
    if uploaded_file is not None:
        cols = st.columns([1.2, 1], gap="large")
        
        with cols[0]:
            st.image(uploaded_file, use_container_width=True, caption="Clinical Preview")
            
        with cols[1]:
            if st.button("Start Clinical Analysis", key="analyze", use_container_width=True):
                with st.spinner("""
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <div class="spinner" style="width: 24px; height: 24px; border-radius: 50%;"></div>
                        Analyzing epidermal patterns...
                    </div>
                """):
                    try:
                        image = Image.open(uploaded_file)
                        processed_image = preprocess_image(image)
                        predictions = model.predict(processed_image)[0]
                        skin_types = ['Dry Skin', 'Acne Skin', 'Oily Skin', 'Normal Skin']
                        max_idx = np.argmax(predictions)
                        confidence = predictions[max_idx] * 100

                        with st.container():
                            st.markdown('<div class="result-card">', unsafe_allow_html=True)
                            
                            # Primary diagnosis
                            st.markdown(f"""
                            <div style="margin-bottom: 2rem;">
                                <div style="color: var(--secondary-text); margin-bottom: 8px;">Primary Diagnosis</div>
                                <div style="font-size: 1.8rem; font-weight: 500; color: var(--on-surface);">
                                    {skin_types[max_idx]}
                                    <span style="color: var(--primary); margin-left: 12px;">{confidence:.1f}%</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Confidence visualization
                            st.markdown(f"""
                            <div style="margin: 2rem 0;">
                                <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
                                    <div style="color: var(--secondary-text);">Diagnostic Confidence</div>
                                    <div style="font-weight: 500;">Clinical Grade</div>
                                </div>
                                <div class="confidence-meter">
                                    <div class="confidence-fill" style="width: {confidence}%"></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Differential diagnosis
                            st.markdown("""
                            <div style="margin-top: 2rem;">
                                <div style="color: var(--secondary-text); margin-bottom: 1.5rem;">Differential Analysis</div>
                                <div class="prediction-grid">
                            """, unsafe_allow_html=True)
                            
                            for i, skin_type in enumerate(skin_types):
                                percent = predictions[i] * 100
                                active_class = "active" if i == max_idx else ""
                                st.markdown(f"""
                                <div class="prediction-card {active_class}">
                                    <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
                                        <div>{skin_type}</div>
                                        <div style="font-weight: 500; color: var(--primary);">{percent:.1f}%</div>
                                    </div>
                                    <div class="confidence-meter">
                                        <div class="confidence-fill" style="width: {percent}%"></div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            st.markdown("""
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Clinical notes
                            st.markdown("""
                            <div style="margin-top: 3rem; padding-top: 2rem; border-top: 1px solid #DFE1E5;">
                                <div style="color: var(--secondary-text); margin-bottom: 1rem;">Clinical Notes</div>
                                <div style="display: grid; gap: 1rem; grid-template-columns: repeat(2, 1fr);">
                                    <div>
                                        <div style="font-size: 0.9rem; color: var(--secondary-text);">Recommendation</div>
                                        <div style="margin-top: 4px;">Consult dermatologist for validation</div>
                                    </div>
                                    <div>
                                        
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown('</div>', unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()