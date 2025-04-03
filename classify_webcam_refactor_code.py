# updated 14
import os
import cv2
from transformers import pipeline
import tensorflow as tf
import pyttsx3
import numpy as np
# from display_test import display
from suggestions import suggesstions

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
predictor = pipeline("text-generation", model="gpt2", max_length=5)
# Load Model Function
def load_model():
    global sess, softmax_tensor, label_lines
    label_lines = [line.rstrip() for line in tf.io.gfile.GFile("logs/output_labels.txt")]
    with tf.io.gfile.GFile("logs/output_graph.pb", 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.compat.v1.import_graph_def(graph_def, name='')
    sess = tf.compat.v1.Session()
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

load_model()

# Function to Predict Sign
def predict(image_data):
    predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    max_score, res = 0.0, ''
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        if score > max_score:
            max_score, res = score, human_string
    return res, max_score

# Initialize Webcam
cap = cv2.VideoCapture(0)
res, score, mem, consecutive, sequence = '', 0.0, '', 0, ''
typed_word = ''
i = 0
engine = pyttsx3.init()
previous_gesture = None
current_suggestions = []
current_index = 0
in_suggestion_mode = True


def predict_next_word(sentence):
    generated_text = predictor(sentence, max_length=10, do_sample=True)[0]['generated_text']
    words = generated_text.split()
    input_length = len(sentence.split())
    new_words = words[input_length:]
    
    return new_words[:3] if new_words else []


def speak(word):
    engine.say(word)
    engine.runAndWait()


def clear_suggestions():
    img_suggestions = np.zeros((400, 500, 3), np.uint8)
    cv2.imshow("Suggestions", img_suggestions)

def toggle_typing_mode():
    global in_suggestion_mode
    in_suggestion_mode = False 
    print("Switched to suggestions mode" if in_suggestion_mode else "Switched to typing mode")

# Function to Display Suggestions
def display_suggestions():
    global sequence, current_index,img_suggestions
    if current_suggestions:
            # Create a black image
            img_suggestions = np.zeros((400, 700, 3), np.uint8)
            start_y = 50

            # Draw each suggestion
            for idx, suggestion in enumerate(current_suggestions):
                color = (0, 255, 0) if idx == current_index else (255, 255, 255)  # Highlight selected word
                cv2.putText(img_suggestions, suggestion, (50, start_y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                start_y += 50

            # Show the suggestions window
            cv2.imshow("Suggestions", img_suggestions)

def update_suggestions(character):
    global current_suggestions, current_index, typed_word

    # typed_word += character
    current_suggestions = suggesstions(character)
    # typed_word_suggestions = suggestions(typed_word) 
    current_index = 0
    display_suggestions()

# def update_suggestions(character):
#     global current_suggestions, current_index, typed_word, sequence,typed_word

#     typed_word += character  # Update the partially typed word

#     # Get suggestions based on the typed word
#     current_suggestions = suggestions(typed_word)  
#     # Get GPT-2 predictions for context-aware suggestions
#     last_words = " ".join(sequence.strip().split()[-3:])  # Use last 3 words for context
#     context_suggestions = predict_next_word(last_words)
#     # Merge both types of suggestions
#     current_suggestions = list(set(typed_word_suggestions + context_suggestions))  # Avoid duplicates
#     current_index = 0

#     display_suggestions()


# def select_suggestion():
#     global current_index, current_suggestions, sequence, in_suggestion_mode, typed_word
#     if current_suggestions:
#         print(f"Selected: {current_suggestions[current_index]}")
#         selected_word = current_suggestions[current_index]
#         if typed_word in selected_word:
#             sequence = sequence[:-len(typed_word) + selected_word]
#             # sequence += current_suggestions[current_index] + ' '
#         else:
#             sequence += selected_word + ' '

#         speak(selected_word)
#         current_suggestions = []
#         current_index = 0
#         typed_word = ""
#         clear_suggestions()

def select_suggestion():
    global current_index, current_suggestions, sequence, typed_word

    if current_suggestions:
        selected_word = current_suggestions[current_index]

        # If typed word exists in the suggestion, replace it
        if typed_word and selected_word.startswith(typed_word):
            sequence = sequence[:-len(typed_word)] + selected_word + ' '
        else:
            sequence += selected_word + ' '

        current_suggestions = []
        current_index = 0
        typed_word = ""
        clear_suggestions()


# Function to Scroll Up
def scroll_up():
    global current_index,img_suggestions
    current_index = (current_index-1)%len(current_suggestions)
    # clear_suggestions()
    display_suggestions()

# Function to Scroll Down
def scroll_down():
    global current_index
    current_index = (current_index+1)%len(current_suggestions)
    # clear_suggestions()
    display_suggestions()

# Main Loop
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
            consecutive = consecutive + 1 if mem == res else 0
            if consecutive == 2 and res not in ['nothing']:
                if res == 'space':
                    if typed_word:
                        speak(typed_word)
                    sequence += ' '
                    typed_word = ''
                elif res == 'del':
                    sequence = sequence[:-1]
                elif res == 'select':
                    select_suggestion()
                elif res == 'scroll up':
                    current_index = (current_index-1)%len(current_suggestions)
                    display_suggestions()
                elif res == 'scroll down':
                    current_index = (current_index+1)%len(current_suggestions)
                    display_suggestions()
                else:
                    if in_suggestion_mode:
                        update_suggestions(res) 
                    else:
                        # update_suggestions(res)
                        sequence += res
                    # typed_word += res
                
                consecutive = 0
        
        i += 1
        cv2.putText(img, f'{res.upper()}', (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
        mem = res
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Initialize Image for Sequence
        img_sequence = np.zeros((200, 1200, 3), np.uint8)
        cv2.putText(img_sequence, sequence.upper(), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Sequence', img_sequence)
        cv2.imshow("Webcam", img)
        
        if cv2.waitKey(1) == 27:
            break

# Cleanup
cv2.destroyAllWindows()
cap.release()
sess.close()





# old
# import os
# import cv2
# from transformers import pipeline
# import tensorflow as tf
# import pyttsx3
# import numpy as np
# # from display_test import display
# from suggestions import suggestions

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# predictor = pipeline("text-generation", model="gpt2", max_length=5)
# # Load Model Function
# def load_model():
#     global sess, softmax_tensor, label_lines
#     label_lines = [line.rstrip() for line in tf.io.gfile.GFile("logs/output_labels.txt")]
#     with tf.io.gfile.GFile("logs/output_graph.pb", 'rb') as f:
#         graph_def = tf.compat.v1.GraphDef()
#         graph_def.ParseFromString(f.read())
#         tf.compat.v1.import_graph_def(graph_def, name='')
#     sess = tf.compat.v1.Session()
#     softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

# load_model()

# # Function to Predict Sign
# def predict(image_data):
#     predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
#     top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
#     max_score, res = 0.0, ''
#     for node_id in top_k:
#         human_string = label_lines[node_id]
#         score = predictions[0][node_id]
#         if score > max_score:
#             max_score, res = score, human_string
#     return res, max_score

# # Initialize Webcam
# cap = cv2.VideoCapture(0)
# res, score, mem, consecutive, sequence = '', 0.0, '', 0, ''
# typed_word = ''
# i = 0
# engine = pyttsx3.init()
# previous_gesture = None
# current_suggestions = []
# current_index = 0
# in_suggestion_mode = True


# def predict_next_word(sentence):
#     generated_text = predictor(sentence, max_length=10, do_sample=True)[0]['generated_text']
#     words = generated_text.split()
#     input_length = len(sentence.split())
#     new_words = words[input_length:]
    
#     return new_words[:3] if new_words else []


# def speak(word):
#     engine.say(word)
#     engine.runAndWait()


# def clear_suggestions():
#     img_suggestions = np.zeros((400, 500, 3), np.uint8)
#     cv2.imshow("Suggestions", img_suggestions)

# def toggle_typing_mode():
#     global in_suggestion_mode
#     in_suggestion_mode = False 
#     print("Switched to suggestions mode" if in_suggestion_mode else "Switched to typing mode")

# # Function to Display Suggestions
# def display_suggestions():
#     global sequence, current_index,img_suggestions
#     if current_suggestions:
#             # Create a black image
#             img_suggestions = np.zeros((400, 700, 3), np.uint8)
#             start_y = 50

#             # Draw each suggestion
#             for idx, suggestion in enumerate(current_suggestions):
#                 color = (0, 255, 0) if idx == current_index else (255, 255, 255)  # Highlight selected word
#                 cv2.putText(img_suggestions, suggestion, (50, start_y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
#                 start_y += 50

#             # Show the suggestions window
#             cv2.imshow("Suggestions", img_suggestions)

# # def update_suggestions(character):
# #     global current_suggestions, current_index, typed_word,typed_word_suggestions

# #     typed_word += character
# #     current_suggestions = suggestions(character)
# #     typed_word_suggestions = suggestions(typed_word) 
# #     current_index = 0
# #     display_suggestions()

# # #  to use
# # def update_suggestions(character):
# #     global current_suggestions, current_index, typed_word, sequence,typed_word,typed_word_suggestions

# #     typed_word += character  # Update the partially typed word

# #     # Get suggestions based on the typed word
# #     current_suggestions = suggestions(typed_word)  
# #     # Get GPT-2 predictions for context-aware suggestions
# #     last_words = " ".join(sequence.strip().split()[-3:])  # Use last 3 words for context
# #     context_suggestions = predict_next_word(last_words)
# #     # Merge both types of suggestions
# #     current_suggestions = list(set(typed_word_suggestions + context_suggestions))  # Avoid duplicates
# #     current_index = 0

# #     display_suggestions()

# # corrected for error

# def update_suggestions(character):
#     global current_suggestions, current_index, typed_word, sequence, typed_word_suggestions

#     typed_word += character  # Update the partially typed word

#     # Get word-based suggestions
#     typed_word_suggestions = suggestions(typed_word)  

#     # Get context-based suggestions using GPT-2
#     last_words = " ".join(sequence.strip().split()[-3:])  # Use last 3 words for context
#     context_suggestions = predict_next_word(last_words)

#     # Merge both types of suggestions (Avoid duplicates)
#     current_suggestions = list(set(typed_word_suggestions + context_suggestions))  

#     current_index = 0
#     display_suggestions()




# # def select_suggestion():
# #     global current_index, current_suggestions, sequence, in_suggestion_mode, typed_word
# #     if current_suggestions:
# #         print(f"Selected: {current_suggestions[current_index]}")
# #         selected_word = current_suggestions[current_index]
# #         if typed_word in selected_word:
# #             sequence = sequence[:-len(typed_word) + selected_word]
# #             # sequence += current_suggestions[current_index] + ' '
# #         else:
# #             sequence += selected_word + ' '

# #         speak(selected_word)
# #         current_suggestions = []
# #         current_index = 0
# #         typed_word = ""
# #         clear_suggestions()

# def select_suggestion():
#     global current_index, current_suggestions, sequence, typed_word

#     if current_suggestions:
#         selected_word = current_suggestions[current_index]

#         # If typed word exists in the suggestion, replace it
#         if typed_word and selected_word.startswith(typed_word):
#             sequence = sequence[:-len(typed_word)] + selected_word + ' '
#         else:
#             sequence += selected_word + ' '

#         current_suggestions = []
#         current_index = 0
#         typed_word = ""
#         clear_suggestions()


# # Function to Scroll Up
# def scroll_up():
#     global current_index,img_suggestions
#     current_index = (current_index-1)%len(current_suggestions)
#     # clear_suggestions()
#     display_suggestions()

# # Function to Scroll Down
# def scroll_down():
#     global current_index
#     current_index = (current_index+1)%len(current_suggestions)
#     # clear_suggestions()
#     display_suggestions()

# # Main Loop
# while True:
#     ret, img = cap.read()
#     img = cv2.flip(img, 1)
#     if ret:
#         x1, y1, x2, y2 = 100, 100, 300, 300
#         img_cropped = img[y1:y2, x1:x2]
#         image_data = cv2.imencode('.jpg', img_cropped)[1].tobytes()
        
#         if i == 4:
#             res_tmp, score = predict(image_data)
#             res = res_tmp
#             i = 0
#             consecutive = consecutive + 1 if mem == res else 0

#             if consecutive == 2 and res not in ['nothing']:
#                 if res == 'space':
#                     if typed_word:
#                         speak(typed_word)
#                     sequence += ' '
#                     typed_word = ''
#                 elif res == 'del':
#                     sequence = sequence[:-1]
#                 elif res == 'select':
#                     select_suggestion()
#                 elif res == 'scroll up':
#                     if current_suggestions:  # Ensure list is not empty
#                         current_index = (current_index - 1) % len(current_suggestions)
#                         display_suggestions()
#                     else:
#                         print("No suggestions available to scroll up.")
#                 elif res == 'scroll down':
#                     if current_suggestions:  # Ensure list is not empty
#                         current_index = (current_index + 1) % len(current_suggestions)
#                         display_suggestions()
#                     else:
#                         print("No suggestions available to scroll down.")
#                 else:
#                     if in_suggestion_mode:
#                         update_suggestions(res) 
#                     else:
#                         # update_suggestions(res)
#                         sequence += res
#                     # typed_word += res
                
#                 consecutive = 0
        
#         i += 1
#         cv2.putText(img, f'{res.upper()}', (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
#         mem = res
#         cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

#         # Initialize Image for Sequence
#         img_sequence = np.zeros((200, 1200, 3), np.uint8)
#         cv2.putText(img_sequence, sequence.upper(), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

#         cv2.imshow('Sequence', img_sequence)
#         cv2.imshow("Webcam", img)
        
#         if cv2.waitKey(1) == 27:
#             break

# # Cleanup
# cv2.destroyAllWindows()
# cap.release()
# sess.close()
