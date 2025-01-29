import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import copy
import cv2
import tensorflow as tf

# Import suggestions module
from suggestions import suggesstions

# Disable TensorFlow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

# Load label file
label_lines = [line.rstrip() for line in tf.io.gfile.GFile("logs/output_labels.txt")]

# Load TensorFlow model
with tf.io.gfile.GFile("logs/output_graph.pb", 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

with tf.compat.v1.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    cap = cv2.VideoCapture(0)
    sequence = ''
    current_suggestions = []
    selected_index = 0  # Index of the currently highlighted suggestion

    while True:
        ret, img = cap.read()
        img = cv2.flip(img, 1)

        if ret:
            x1, y1, x2, y2 = 100, 100, 300, 300
            img_cropped = img[y1:y2, x1:x2]
            image_data = cv2.imencode('.jpg', img_cropped)[1].tobytes()

            res_tmp, score = predict(image_data)
            res = res_tmp.upper()

            # Add recognized character to the sequence or trigger actions
            if res not in ['NOTHING', 'SPACE', 'DEL', 'SCROLL_UP', 'SCROLL_DOWN', 'SELECT']:
                sequence += res
                current_suggestions = suggesstions(res)  # Get word suggestions for the recognized character
                selected_index = 0  # Reset to the first suggestion
            elif res == 'SCROLL_UP' and current_suggestions:
                selected_index = (selected_index - 1) % len(current_suggestions)
            elif res == 'SCROLL_DOWN' and current_suggestions:
                selected_index = (selected_index + 1) % len(current_suggestions)
            elif res == 'SELECT' and current_suggestions:
                sequence += current_suggestions[selected_index] + ' '
                current_suggestions = []
                selected_index = 0

            # Display recognized character and suggestions
            cv2.putText(img, f"Character: {res} (score: {score:.2f})", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if current_suggestions:
                for i, suggestion in enumerate(current_suggestions):
                    color = (0, 255, 0) if i == selected_index else (255, 255, 255)
                    cv2.putText(img, suggestion, (10, 100 + i * 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Display sequence
            cv2.putText(img, f"Sequence: {sequence}", (10, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.imshow("Sign Language Recognition", img)

            if cv2.waitKey(1) & 0xFF == 27:  # Escape key
                break

cap.release()
cv2.destroyAllWindows()
