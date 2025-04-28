import os
import uuid
import cv2
import pytesseract
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static/outputs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Make sure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Set tesseract path if needed
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"

# Store history in memory
detection_history = []

@app.route('/')
def home():
    return render_template('index.html', history=detection_history)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['image']
    filename = secure_filename(file.filename)
    unique_id = str(uuid.uuid4())
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_id + "_" + filename)
    file.save(input_path)

    img = cv2.imread(input_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(blur, 30, 200)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    plate = None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    for c in contours:
        approx = cv2.approxPolyDP(c, 0.018 * cv2.arcLength(c, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(c)
            plate = gray[y:y+h, x:x+w]
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            break

    result = {'text': 'Plate not found', 'images': []}

    # Save and collect image URLs
    def save_image(image, label):
        out_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{unique_id}_{label}.jpg")
        cv2.imwrite(out_path, image)
        result['images'].append(f"/{out_path.replace(os.sep, '/')}")
        return out_path

    save_image(img, 'original')
    save_image(gray, 'gray')
    save_image(edged, 'edged')

    if plate is not None:
        save_image(plate, 'plate')
        config = '-l eng --oem 3 --psm 8'
        text = pytesseract.image_to_string(plate, config=config).strip()
        result['text'] = text

    # Append to history
    detection_history.insert(0, {
        'id': unique_id,
        'text': result['text'],
        'images': result['images']
    })

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
