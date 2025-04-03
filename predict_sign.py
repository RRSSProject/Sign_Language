# import sys
# import os
# import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
# import copy
# import cv2
# import tensorflow as tf

# # Import suggestions module
# from suggestions import suggesstions

# # Disable TensorFlow compilation warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# def predict(image_data):
#     predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
#     top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
#     max_score = 0.0
#     res = ''
#     for node_id in top_k:
#         human_string = label_lines[node_id]
#         score = predictions[0][node_id]
#         if score > max_score:
#             max_score = score
#             res = human_string
#     return res, max_score

# # Load label file
# label_lines = [line.rstrip() for line in tf.io.gfile.GFile("logs/output_labels.txt")]

# # Load TensorFlow model
# with tf.io.gfile.GFile("logs/output_graph.pb", 'rb') as f:
#     graph_def = tf.compat.v1.GraphDef()
#     graph_def.ParseFromString(f.read())
#     _ = tf.import_graph_def(graph_def, name='')

# with tf.compat.v1.Session() as sess:
#     softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
#     cap = cv2.VideoCapture(0)
#     sequence = ''
#     current_suggestions = []
#     selected_index = 0  # Index of the currently highlighted suggestion

#     while True:
#         ret, img = cap.read()
#         img = cv2.flip(img, 1)

#         if ret:
#             x1, y1, x2, y2 = 100, 100, 300, 300
#             img_cropped = img[y1:y2, x1:x2]
#             image_data = cv2.imencode('.jpg', img_cropped)[1].tobytes()

#             res_tmp, score = predict(image_data)
#             res = res_tmp.upper()

#             # Add recognized character to the sequence or trigger actions
#             if res not in ['NOTHING', 'SPACE', 'DEL', 'SCROLL_UP', 'SCROLL_DOWN', 'SELECT']:
#                 sequence += res
#                 current_suggestions = suggesstions(res)  # Get word suggestions for the recognized character
#                 selected_index = 0  # Reset to the first suggestion
#             elif res == 'SCROLL_UP' and current_suggestions:
#                 selected_index = (selected_index - 1) % len(current_suggestions)
#             elif res == 'SCROLL_DOWN' and current_suggestions:
#                 selected_index = (selected_index + 1) % len(current_suggestions)
#             elif res == 'SELECT' and current_suggestions:
#                 sequence += current_suggestions[selected_index] + ' '
#                 current_suggestions = []
#                 selected_index = 0

#             # Display recognized character and suggestions
#             cv2.putText(img, f"Character: {res} (score: {score:.2f})", (10, 50),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#             if current_suggestions:
#                 for i, suggestion in enumerate(current_suggestions):
#                     color = (0, 255, 0) if i == selected_index else (255, 255, 255)
#                     cv2.putText(img, suggestion, (10, 100 + i * 30),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

#             # Display sequence
#             cv2.putText(img, f"Sequence: {sequence}", (10, 300),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

#             cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
#             cv2.imshow("Sign Language Recognition", img)

#             if cv2.waitKey(1) & 0xFF == 27:  # Escape key
#                 break

# cap.release()
# cv2.destroyAllWindows()

# import sys
# import os
# import numpy as np
# import cv2
# import tensorflow.compat.v1 as tf  # Use TensorFlow 1.x compatibility mode
# tf.disable_v2_behavior()  # Disable TensorFlow 2 behavior

# # Import suggestions module
# from suggestions import suggesstions

# # Disable TensorFlow compilation warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# # Load the frozen graph
# def load_frozen_graph(model_filename):
#     with tf.io.gfile.GFile(model_filename, "rb") as f:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(f.read())

#     with tf.Graph().as_default() as graph:
#         tf.import_graph_def(graph_def, name="")
#     return graph

# # Load the trained model
# MODEL_PATH = "logs/output_graph.pb"
# LABEL_PATH = "logs/output_labels.txt"

# # Check if model exists
# if not os.path.exists(MODEL_PATH):
#     raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
# if not os.path.exists(LABEL_PATH):
#     raise FileNotFoundError(f"Label file not found: {LABEL_PATH}")

# # Load label file
# label_lines = [line.rstrip() for line in tf.io.gfile.GFile(LABEL_PATH)]

# # Load graph and start session
# graph = load_frozen_graph(MODEL_PATH)
# sess = tf.Session(graph=graph)

# # Get input and output tensors
# input_tensor = sess.graph.get_tensor_by_name("DecodeJpeg/contents:0")
# output_tensor = sess.graph.get_tensor_by_name("final_result:0")

# # Function to predict the sign
# def predict(image_data):
#     predictions = sess.run(output_tensor, {input_tensor: image_data})
#     top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
#     max_score = 0.0
#     res = ''
#     for node_id in top_k:
#         human_string = label_lines[node_id]
#         score = predictions[0][node_id]
#         if score > max_score:
#             max_score = score
#             res = human_string
#     return res, max_score

# # Initialize video capture
# cap = cv2.VideoCapture(0)
# sequence = ''
# current_suggestions = []
# selected_index = 0  # Index of the currently highlighted suggestion

# while True:
#     ret, img = cap.read()
#     img = cv2.flip(img, 1)  # Flip horizontally for better usability

#     if ret:
#         # Define region of interest for sign recognition
#         x1, y1, x2, y2 = 100, 100, 300, 300
#         img_cropped = img[y1:y2, x1:x2]
#         image_data = cv2.imencode('.jpg', img_cropped)[1].tobytes()

#         # Predict the sign
#         res_tmp, score = predict(image_data)
#         res = res_tmp.upper()

#         # Handle recognized gestures
#         if res not in ['NOTHING', 'SPACE', 'DEL', 'SCROLL_UP', 'SCROLL_DOWN', 'SELECT']:
#             sequence += res
#             current_suggestions = suggesstions(res)  # Get word suggestions
#             selected_index = 0  # Reset selection
#         elif res == 'DEL' and len(sequence) > 0:
#             sequence = sequence[:-1]  # Remove the last letter
#         elif res == 'SCROLL_UP' and current_suggestions:
#             selected_index = (selected_index - 1) % len(current_suggestions)
#         elif res == 'SCROLL_DOWN' and current_suggestions:
#             selected_index = (selected_index + 1) % len(current_suggestions)
#         elif res == 'SELECT' and current_suggestions:
#             sequence += current_suggestions[selected_index] + ' '
#             current_suggestions = []
#             selected_index = 0

#         # Display recognized character and confidence score
#         cv2.putText(img, f"Character: {res} (score: {score:.2f})", (10, 50),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

#         # Display word suggestions
#         if current_suggestions:
#             for i, suggestion in enumerate(current_suggestions):
#                 color = (0, 255, 0) if i == selected_index else (255, 255, 255)
#                 cv2.putText(img, suggestion, (10, 100 + i * 30),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

#         # Display recognized word sequence
#         cv2.putText(img, f"Sequence: {sequence}", (10, 300),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

#         # Draw the detection box
#         cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
#         cv2.imshow("Sign Language Recognition", img)

#         # Press 'Esc' to exit
#         if cv2.waitKey(1) & 0xFF == 27:
#             break

# cap.release()
# cv2.destroyAllWindows()

import sys
import os
import numpy as np
import cv2
import tensorflow as tf  # TensorFlow 2.x

# Import suggestions module
from suggestions import suggesstions

# Disable TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load the trained model
MODEL_PATH = "logs/output_graph.pb"
LABEL_PATH = "logs/output_labels.txt"

# Check if model exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
if not os.path.exists(LABEL_PATH):
    raise FileNotFoundError(f"Label file not found: {LABEL_PATH}")

# Load label file
label_lines = [line.rstrip() for line in open(LABEL_PATH)]

# Load the trained TensorFlow 2.x model
model = tf.saved_model.load(MODEL_PATH)
predict_fn = model.signatures["serving_default"]

# Function to predict the sign
def predict(image_data):
    image_tensor = tf.convert_to_tensor(image_data, dtype=tf.uint8)
    image_tensor = tf.image.decode_jpeg(image_tensor, channels=3)
    image_tensor = tf.image.resize(image_tensor, [224, 224]) / 255.0
    image_tensor = tf.expand_dims(image_tensor, axis=0)
    
    predictions = predict_fn(tf.constant(image_tensor))
    predictions = list(predictions.values())[0].numpy()[0]
    
    max_score = np.max(predictions)
    res = label_lines[np.argmax(predictions)]
    return res.upper(), max_score

# Initialize video capture
cap = cv2.VideoCapture(0)
sequence = ''
current_suggestions = []
selected_index = 0  # Index of the currently highlighted suggestion

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)  # Flip for better usability

    if ret:
        # Define region of interest for sign recognition
        x1, y1, x2, y2 = 100, 100, 300, 300
        img_cropped = img[y1:y2, x1:x2]
        image_data = cv2.imencode('.jpg', img_cropped)[1].tobytes()

        # Predict the sign
        res_tmp, score = predict(image_data)
        res = res_tmp.upper()

        # Handle recognized gestures
        if res not in ['NOTHING', 'SPACE', 'DEL', 'SCROLL_UP', 'SCROLL_DOWN', 'SELECT']:
            sequence += res
            current_suggestions = suggesstions(res)  # Get word suggestions
            selected_index = 0  # Reset selection
        elif res == 'DEL' and len(sequence) > 0:
            sequence = sequence[:-1]  # Remove the last letter
        elif res == 'SCROLL_UP' and current_suggestions:
            selected_index = (selected_index - 1) % len(current_suggestions)
        elif res == 'SCROLL_DOWN' and current_suggestions:
            selected_index = (selected_index + 1) % len(current_suggestions)
        elif res == 'SELECT' and current_suggestions:
            sequence += current_suggestions[selected_index] + ' '
            current_suggestions = []
            selected_index = 0

        # Display recognized character and confidence score
        cv2.putText(img, f"Character: {res} (score: {score:.2f})", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display word suggestions
        if current_suggestions:
            for i, suggestion in enumerate(current_suggestions):
                color = (0, 255, 0) if i == selected_index else (255, 255, 255)
                cv2.putText(img, suggestion, (10, 100 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Display recognized word sequence
        cv2.putText(img, f"Sequence: {sequence}", (10, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw the detection box
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.imshow("Sign Language Recognition", img)

        # Press 'Esc' to exit
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
