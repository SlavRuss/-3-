from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

app = Flask(__name__)

app.config.update({
    'upload_folder': 'static/uploads',
    'class_images': 'static/images',
    'allowed_extensions': {'png', 'jpg', 'jpeg', 'gif'},
    'max_content_length': 16 * 1024 * 1024,
    'model_path': 'cat_classifier_model.keras',
    'image_size': (256, 256)
})

cat_classes = {
    0: {'name': 'Бенгальская кошка', 'description': 'Экзотическая кошка с диким окрасом', 'image': 'Бенгальская кошка.jpg'},
    1: {'name': 'Ориентальная кошка', 'description': 'Утонченная и элегантная', 'image': 'Ориентальная кошка.jpg'},
    2: {'name': 'Русская голубая', 'description': 'Аристократичная с серебристой шерстью', 'image': 'Русская голубая.jpg'},
    3: {'name': 'Сноу-шу', 'description': 'Кошки в "носочках" с необычным окрасом', 'image': 'Сноу-шу.jpg'},
    4: {'name': 'Дворовая кошка', 'description': 'Уникальная и неповторимая', 'image': 'Дворовая кошка.jpg'},
    5: {'name': 'Абиссинская кошка', 'description': 'Королевская осанка и дикий окрас', 'image': 'Абиссинская кошка.jpg'},
    6: {'name': 'Британская кошка', 'description': 'Плюшевая и флегматичная', 'image': 'Британская кошка.jpg'},
    7: {'name': 'Мейн-кун', 'description': 'Гигантский и пушистый', 'image': 'Мейн-кун.jpg'},
    8: {'name': 'Шотландская вислоухая', 'description': 'С милыми сложенными ушками', 'image': 'Шотландская вислоухая.jpg'},
    9: {'name': 'Сфинкс', 'description': 'Лысый и экстравагантный', 'image': 'Сфинкс.jpg'}
}

model = load_model(app.config['model_path'])
model.make_predict_function()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['allowed_extensions']

def preprocess_image(image_path):
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(app.config['image_size'])
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route('/')
def home():
    return render_template('index.html',
                           cat="Угадай кота",
                           page="home")

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['upload_folder'], filename)
        file.save(filepath)

        try:
            processed_img = preprocess_image(filepath)
            predictions = model.predict(processed_img)
            class_id = np.argmax(predictions[0])
            confidence = round(float(predictions[0][class_id]) * 100, 2)

            class_info = cat_classes[class_id]

            result = {
                'name': class_info['name'],
                'description': class_info['description'],
                'confidence': confidence,
                'user_image': url_for('static', filename=f'uploads/{filename}'),
                'class_image': url_for('static', filename=f'images/{class_info["image"]}')
            }
        except Exception as e:
            print(f"Ошибка при предсказании: {e}")
            result = None

        return render_template('index.html',
                               cat="Результат",
                               page="result",
                               result=result)
    return redirect(request.url)

@app.context_processor
def inject_globals():
    return {
        'css_url': url_for('static', filename='css/cat.css'),
        'js_url': url_for('static', filename='js/cat.js'),
        'default_image': url_for('static', filename='images/proto-cat.png')
    }

@app.route('/reload')
def reload():
    upload_folder = app.config['upload_folder']
    for filename in os.listdir(upload_folder):
        file_path = os.path.join(upload_folder, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f'Ошибка при удалении {file_path}: {e}')

    return redirect(url_for('home'))

@app.route('/upload_ajax', methods=['POST'])
def upload_ajax():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['upload_folder'], filename)
        file.save(filepath)

        try:
            processed_img = preprocess_image(filepath)
            predictions = model.predict(processed_img)
            class_id = np.argmax(predictions[0])
            confidence = round(float(predictions[0][class_id]) * 100, 2)

            class_info = cat_classes[class_id]

            result = {
                'name': class_info['name'],
                'description': class_info['description'],
                'confidence': confidence,
                'user_image': url_for('static', filename=f'uploads/{filename}'),
                'class_image': url_for('static', filename=f'images/{class_info["image"]}')
            }
        except Exception as e:
            print(f"Ошибка при предсказании: {e}")
            result = None
        return render_template('result_partial.html', result=result)
    return jsonify({'error': 'File type not allowed'}), 400

if __name__ == '__main__':
    required_folders = [
        app.config['upload_folder'],
        app.config['class_images'],
        'static/css',
        'static/js',
        'templates'
    ]
    for folder in required_folders:
        os.makedirs(folder, exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
