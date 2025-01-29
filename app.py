import os
import cv2
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
import numpy as np

# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load TensorFlow model
label_lines = [line.rstrip() for line in tf.io.gfile.GFile("logs/output_labels.txt")]

with tf.io.gfile.GFile("logs/output_graph.pb", 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

with tf.compat.v1.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    # Video capture
    cap = cv2.VideoCapture(0)

    def predict(image_data):
        predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        max_score = 0.0
        res = ''
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            if score > max_score:
                max_score = score
                res = human_string
        return res, max_score

    # Handle video streaming and predictions in real-time
    def gen_frames():
        res, score = '', 0.0
        mem = ''
        consecutive = 0
        sequence = ''
        i = 0

        while True:
            ret, img = cap.read()
            img = cv2.flip(img, 1)
            
            if ret:
                x1, y1, x2, y2 = 100, 100, 300, 300
                img_cropped = img[y1:y2, x1:x2]

                image_data = cv2.imencode('.jpg', img_cropped)[1].tobytes()

                if i == 4:
                    res_tmp, score = predict(image_data)
                    res = res_tmp
                    i = 0
                    if mem == res:
                        consecutive += 1
                    else:
                        consecutive = 0
                    if consecutive == 2 and res not in ['nothing']:
                        if res == 'space':
                            sequence += ' '
                        elif res == 'del':
                            sequence = sequence[:-1]
                        else:
                            sequence += res
                        consecutive = 0
                i += 1

                cv2.putText(img, '%s' % (res.upper()), (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 4)
                cv2.putText(img, '(score = %.5f)' % (float(score)), (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
                mem = res

                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # Encode image as JPEG and return as response
                ret, buffer = cv2.imencode('.jpg', img)
                frame = buffer.tobytes()

                # Yield the frame for streaming
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
                

    # Video feed route for frontend
    @app.route('/video-feed')
    def video_feed():
        return Response(gen_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    # Prediction endpoint to return the current predicted sign and sequence
    @app.route('/predict', methods=['GET'])
    def predict_sign():
        return jsonify({'prediction': res, 'sequence': sequence})

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
