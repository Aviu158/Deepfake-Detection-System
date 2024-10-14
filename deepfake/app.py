# app.py
from flask import Flask, request, jsonify, render_template
from deepfake_model import DeepfakeModel
import os

app = Flask(__name__)
model = DeepfakeModel()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        prediction = model.predict(file_path)
        return jsonify({'prediction': prediction})

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
