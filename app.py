from flask import Flask, request, render_template, Response
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import io
import base64
import cv2

app = Flask(__name__)

# Define the path to your model
model_path = os.path.join('models', 'animal_detection_model.keras')
model = tf.keras.models.load_model(model_path)
class_names = ['cow', 'dog']

def preprocess_image(image):
    # Convert image to RGB if it's not already
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return class_names[int(prediction[0][0] > 0.5)]

def generate_frames(video_path):
    # Open video file
    cap = cv2.VideoCapture(video_path)
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Convert frame to PIL image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Prediction
        label = predict(image)
        
        # Draw label on frame
        cv2.putText(frame, f'Prediction: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_from_path():
    video_path = request.form.get('video_path')
    
    if not video_path or not os.path.exists(video_path):
        return 'Invalid video path or file does not exist.'

    try:
        return Response(generate_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        return f'Error processing video: {e}'

if __name__ == '__main__':
    app.run(debug=True)
