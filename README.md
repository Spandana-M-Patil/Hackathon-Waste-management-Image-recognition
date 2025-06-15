This is the starter for our hackathon project.
Current status:
1. Data screening is done. Moving onto training.
2. Done with training it's giving around 60% accuracy and trying to generalize the model to fit real world data.
3. Fine-tuned the model to fit real world data, trained model now giving around 80% accuracy. 
4. Successfully integrated the trained model into the frontend using Streamlit, enabling real-time waste classification through an interactive web interface.

# Smart Garbage Segregation App

A **Streamlit web app** that uses a **TensorFlow image classification model** to detect different types of garbage and guide users on how to dispose of it properly.

## Demo

Upload an image of garbage (e.g., a bottle, can, paper, etc.) and get:
- The **predicted class** (like plastic, paper, metal, etc.)
- A **friendly disposal guide**!

## Supported Categories

The model can classify garbage into:
- `cardboard`
- `glass`
- `metal`
- `paper`
- `plastic`
- `trash` (non-recyclable)

## How It Works

1. Upload an image (JPG, PNG, or JPEG).
2. Image is resized to 224x224 and normalized.
3. Passed to a **pre-trained TensorFlow model** stored in the `trashClassifier` folder.
4. The model predicts the garbage type.
5. The app displays the result and a short guide.

## Project Structure

smart-garbage-segregation/
├── trashClassifier/ # Saved TensorFlow model (SavedModel format)
├── app.py # Main Streamlit app file
├── requirements.txt 
└── README.md

## Model Format

Model is loaded using:
```python
tf.keras.layers.TFSMLayer("trashClassifier", call_endpoint="serving_default")
```
## Setup & Run

### 1. Clone the Repo
```bash
git clone https://github.com/your-username/smart-garbage-segregation.git
cd smart-garbage-segregation
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Run the App
```bash
streamlit run app.py
```




