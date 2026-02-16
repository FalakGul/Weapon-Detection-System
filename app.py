# app.py
from flask import Flask, request, jsonify, send_from_directory
import os
import cv2
from ultralytics import YOLO

app = Flask(__name__, static_folder='static')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'static/results'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

model = YOLO('weapon_best.pt')  # your trained model

@app.route('/')
def index():
    return app.send_static_file('weapon-ui.html')

# IMAGE DETECTION
@app.route('/detect/image', methods=['POST'])
def detect_image():
    file = request.files['file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    img = cv2.imread(filepath)
    results = model(img, conf=0.5)[0]
    detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0].item()
        cls = int(box.cls[0].item())
        label = model.names[cls]
        h, w = img.shape[:2]
        detections.append({
            "label": label,
            "score": round(conf, 2),
            "xmin": x1/w, "ymin": y1/h,
            "xmax": x2/w, "ymax": y2/h
        })
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4)
        cv2.putText(img, f"{label} {conf:.1%}", (x1, y1-10),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 3)

    output_path = os.path.join(app.config['RESULT_FOLDER'], 'result_image.jpg')
    cv2.imwrite(output_path, img)
    return jsonify({"detections": detections})

# VIDEO DETECTION – FULLY WORKING
@app.route('/detect/video', methods=['POST'])
def detect_video():
    file = request.files['file']
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    output_path = os.path.join(app.config['RESULT_FOLDER'], 'result_video.mp4')
    file.save(input_path)

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'H264'), fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret: break
        results = model(frame, conf=0.5)[0]
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            label = model.names[cls]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
            cv2.putText(frame, f"{label} {conf:.1%}", (x1, y1-10),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)
        out.write(frame)

    cap.release()
    out.release()
    return jsonify({"video_url": "/results/result_video.mp4"})

@app.route('/results/<filename>')
def results(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
    print("WEAPON DETECTION SYSTEM → http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)