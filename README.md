# Plant Disease Detection

A deep learning web application that detects plant diseases from leaf images using MobileNetV2 Transfer Learning.

## Dataset

PlantVillage Dataset

## Features

- Detects 15 different plant disease classes
- Built using MobileNetV2 Transfer Learning
- User-friendly web interface with Streamlit
- Upload a leaf image and get an instant prediction

## Technologies Used

- Python
- TensorFlow / Keras
- MobileNetV2
- Streamlit
- NumPy
- Pillow
- Scikit-learn

## Model Performance

- Validation Accuracy: 93%
- 15 Disease Classes

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/SwathiAdireddy/plant-disease-detection.git
cd plant-disease-detection
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

## Project Structure

```text
plant-disease-detection/
│
├── app.py
├── plant_disease_mobilenet.keras
├── requirements.txt
├── README.md
└── Plant_Disease_Detection.ipynb
```

## Classes Detected

- Pepper__bell___Bacterial_spot
- Pepper__bell___healthy
- Potato___Early_blight
- Potato___Late_blight
- Potato___healthy
- Tomato_Bacterial_spot
- Tomato_Early_blight
- Tomato_Late_blight
- Tomato_Leaf_Mold
- Tomato_Septoria_leaf_spot
- Tomato_Spider_mites_Two_spotted_spider_mite
- Tomato__Target_Spot
- Tomato__Tomato_YellowLeaf__Curl_Virus
- Tomato__Tomato_mosaic_virus
- Tomato_healthy

## Author

Swathi
