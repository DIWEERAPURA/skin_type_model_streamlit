# Advanced Skin Analysis Platform

A clinical-grade skin analysis web application powered by deep learning. This project leverages a TensorFlow model to classify dermatological images into skin type categories with high diagnostic confidence. It demonstrates expertise in machine learning, computer vision, and modern web development practices.

## Overview

This application uses the pre-trained model `skin_type_model_final.h5` to analyze skin images uploaded by users. The model predicts one of the following skin conditions:

- **Dry Skin**
- **Acne Skin**
- **Oily Skin**
- **Normal Skin**

The results include a primary diagnosis along with diagnostic confidence levels and differential analysis, providing an intuitive interface for clinicians and dermatology enthusiasts.

## Features

- **Deep Learning Integration:** Utilizes TensorFlow to load and run predictions on a sophisticated Keras model.
- **Interactive Web UI:** Built with [Streamlit](https://streamlit.io) for a seamless and modern user experience.
- **Responsive Design:** Includes advanced CSS styling with Material Design influences to create a professional interface.
- **Real-time Analysis:** Processes images on the fly with a user-friendly upload interface and provides immediate analytical feedback.
- **Clinical-Grade Results:** Displays detailed results with confidence meters and differential diagnosis for multiple skin conditions.

## Technologies Used

- **Python 3**
- **TensorFlow & Keras**
- **Streamlit**
- **Pillow (PIL)**
- **NumPy**

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/skin_type_model_mine.git
   cd skin_type_model_mine
   ```

2. **Set up a virtual environment and install dependencies:**

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
pip install -r requirements.txt
```

3. **Place your Keras model:**

Ensure that the skin_type_model_final.h5 file is located in the project root or update the path in app.py accordingly.

4. **Run the application:**

```
streamlit run app.py
```

## Usage

- **Upload Image**: Drag and drop or browse to upload a dermatological image.
- **Start Analysis**: Click the "Start Clinical Analysis" button to perform skin type prediction.
- **Review Results**: View the primary diagnosis, diagnostic confidence, and differential diagnosis.

## Skill Demostrated

- **Machine Learning & Deep Learning**: Developed and integrated a custom model for skin type classification.
- **Web Development**: Built a responsive and modern web interface using Streamlit and custom CSS.
- **Data Processing**: Implemented efficient image pre-processing routines using Pillow and NumPy.
- **Software Engineering**: Applied best practices including modular code structure, error handling, and performance optimizations.
- **UI/UX Design**: Designed an intuitive and engaging user interface that blends technical functionality with clinical aesthetics.

## License

This project is licensed under the MIT License.

## Acknowledgments

<small>Thanks to the open-source community for providing invaluable tools and libraries.
Inspired by modern clinical imaging applications and the latest trends in dermatological diagnostics.</small>
