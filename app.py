import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, Response, request, redirect, url_for, session, jsonify, flash
from flask_cors import CORS
import time
import mysql.connector
import re
from datetime import datetime,timedelta
import requests

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'secret_key'
CORS(app)

# MySQL Database Configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'sign_language_db'
}

# Global variables for recognition
video_active = False
latest_prediction = {"res": "NOTHING", "score": 0.0}
sequence = ""
suggestions_list = []
current_index = 0
last_gesture_time = time.time()

# Helper functions
def validate_password(password):
    if len(password) < 8:
        return False
    if not re.search("[A-Z]", password):
        return False
    if not re.search("[0-9]", password):
        return False
    if not re.search("[!@#$%^&*()_+]", password):
        return False
    return True

def get_db_connection():
    return mysql.connector.connect(**db_config)

# Load TensorFlow model
label_lines = [line.rstrip() for line in tf.io.gfile.GFile("logs/output_labels.txt")]

graph = tf.Graph()
with graph.as_default():
    with tf.io.gfile.GFile("logs/output_graph.pb", 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

sess = tf.compat.v1.Session(graph=graph)
softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

def suggestions(character):
    word_dict = {
        "A": ["Available", "Appreciate", "Awesome", "All the best", "At your service"],
        "B": ["Best regards", "Busy now", "Be right back", "By the way", "Back soon"],
        "C": ["Call me", "Catch you later", "Cheers", "Can we talk?", "Cool"],
        "D": ["Do not disturb", "Done", "Definitely", "Don't worry", "Don't forget"],
        "E": ["Excellent", "Excuse me", "Everything is fine", "Easy", "Enjoy"],
        "F": ["For your information", "Fine", "Fantastic", "Feel free", "Follow up"],
        "G": ["Good morning", "Great", "Got it", "Good evening", "Good luck"],
        "H": ["Hi", "Hello", "Hope you’re doing well", "How are you?", "Have a great day"],
        "I": ["I’m here", "I agree", "In progress", "I’m available", "I’m sorry"],
        "J": ["Just a moment", "Just checking in", "Join me", "Just kidding", "Jump in"],
        "K": ["Keep going", "Know this", "Keep me posted", "Kind regards", "Keep it up"],
        "L": ["Let me know", "Looking forward", "Let’s go", "Long time no see", "Love it"],
        "M": ["Meet me", "Much appreciated", "My bad", "Miss you", "More details"],
        "N": ["No problem", "Noted", "Next time", "No worries", "Nice to meet you"],
        "O": ["Okay", "On my way", "Oh no", "Out now", "One moment"],
        "P": ["Please", "Perfect", "Pretty good", "Please confirm", "Proceed"],
        "Q": ["Quick question", "Quite busy", "Questions?", "Quiet now", "Quickly done"],
        "R": ["Right away", "Relax", "Rest assured", "Reach out", "Really good"],
        "S": ["See you", "Sounds good", "Sure", "Stay safe", "So sorry"],
        "T": ["Thank you", "Talk later", "Take care", "The best", "That’s fine"],
        "U": ["Understood", "Unbelievable", "Up to you", "Urgent", "Until later"],
        "V": ["Very good", "View it", "Virtually there", "Very well", "Verify"],
        "W": ["What’s up", "Will do", "Well done", "Welcome", "Work in progress"],
        "X": ["XOXO", "eXcellent idea", "eXactly", "eXpect more", "eXtra help"],
        "Y": ["Yes", "You’re welcome", "You too", "Your call", "Yes please"],
        "Z": ["Zoom meeting", "Zero worries", "Zip it", "Zeal", "Zoned out"]
    }
    return word_dict.get(character.upper(), [])


def predict(image_data):
    global latest_prediction, sequence, suggestions_list, current_index, last_gesture_time
    predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

    max_score = 0.0
    res = "NOTHING"
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        if score > max_score:
            max_score = score
            res = human_string

    latest_prediction["res"] = res
    latest_prediction["score"] = max_score

    if time.time() - last_gesture_time >= 5:
        if res == "del":
            sequence = sequence[:-1]
        elif res == "space":
            sequence += " "
        elif res == "select":
            if suggestions_list and current_index < len(suggestions_list):
                sequence += suggestions_list[current_index] + " "
                suggestions_list = []
        elif res == "scroll up":
            current_index = max(0, current_index - 1)
        elif res == "scroll down":
            current_index = min(len(suggestions_list) - 1, current_index + 1)
        elif res not in ["nothing", "scroll up", "scroll down", "select", "space", "del"]:
            sequence += res
            suggestions_list = suggestions(res)
            current_index = 0

        last_gesture_time = time.time()

    return res, max_score

def gen_frames():
    global video_active
    cap = cv2.VideoCapture(0) if video_active else None

    while video_active and cap and cap.isOpened():
        ret, img = cap.read()
        img = cv2.flip(img, 1)

        if ret:
            x1, y1, x2, y2 = 100, 100, 300, 300
            img_cropped = img[y1:y2, x1:x2]
            img_resized = cv2.resize(img_cropped, (224, 224))
            image_data = cv2.imencode('.jpg', img_resized)[1].tobytes()

            res, score = predict(image_data)
            # cv2.putText(img, f'{res.upper()}', (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            # cv2.putText(img, f'Score: {score:.5f}', (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    if cap:
        cap.release()

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        username = request.form['username']
        gender = request.form['gender']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash("Passwords don't match!", 'danger')
            return redirect(url_for('signup'))
        
        if not validate_password(password):
            flash("Password must be at least 8 characters with one uppercase, one number and one special character", 'danger')
            return redirect(url_for('signup'))

        try:
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)
            
            # Check if username or email exists
            cursor.execute("SELECT * FROM users WHERE username = %s OR email = %s", (username, email))
            if cursor.fetchone():
                flash("Username or email already exists!", 'danger')
                return redirect(url_for('signup'))
            
            # Insert user
            cursor.execute("""
                INSERT INTO users (name, username, gender, email, password, created_at)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (name, username, gender, email, password, datetime.now()))
            
            conn.commit()
            flash("Account created successfully! Please login.", 'success')
            return redirect(url_for('login'))
            
        except mysql.connector.Error as err:
            flash(f"Database error: {err}", 'danger')
        finally:
            cursor.close()
            conn.close()
    
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username_or_email = request.form['username_or_email']
        password = request.form['password']

        try:
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)
            
            cursor.execute("""
                SELECT * FROM users 
                WHERE (username = %s OR email = %s) AND password = %s
            """, (username_or_email, username_or_email, password))
            
            user = cursor.fetchone()
            
            if user:
                session['user'] = {
                    'id': user['id'],
                    'name': user['name'],
                    'username': user['username'],
                    'email': user['email'],
                    'gender': user['gender'],
                    'created_at': user['created_at'].strftime('%B %d, %Y')
                }
                return redirect(url_for('dashboard'))
            
            flash("Invalid credentials!", 'danger')
        except mysql.connector.Error as err:
            flash(f"Database error: {err}", 'danger')
        finally:
            cursor.close()
            conn.close()
    
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', user=session['user'])

@app.route('/start_recognition')
def start_recognition():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    global video_active, sequence, suggestions_list, current_index
    video_active = True
    sequence = ""  # Reset sequence when starting
    suggestions_list = []  # Reset suggestions
    current_index = 0  # Reset index
    return redirect(url_for('recognition'))

@app.route('/recognition')
def recognition():
    if 'user' not in session:
        return redirect(url_for('login'))
    if not video_active:
        return redirect(url_for('dashboard'))
    global sequence, suggestions_list, current_index
    sequence = ""  # Reset sequence on page load
    suggestions_list = []
    current_index = 0
    
    return render_template('recognition.html')

@app.route('/exit_recognition')
def exit_recognition():
    global video_active, sequence, suggestions_list, current_index
    video_active = False
    sequence = ""  # Clear the sequence
    suggestions_list = []  # Clear suggestions
    current_index = 0  # Reset index
    return redirect(url_for('dashboard'))

@app.route('/logout')
def logout():
    global video_active, sequence, suggestions_list
    video_active = False
    sequence = ""
    suggestions_list = []
    session.pop('user', None)
    return redirect(url_for('home'))

# API endpoints
@app.route('/video-feed')
def video_feed():
    if 'user' not in session or not video_active:
        return "", 204
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/update_profile', methods=['POST'])
def update_profile():
    if 'user' not in session:
        return jsonify({'success': False, 'message': 'Not logged in'}), 401

    data = request.get_json()
    user_id = session['user']['id']

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if username is already taken by another user
        cursor.execute("SELECT id FROM users WHERE username = %s AND id != %s", 
                      (data['username'], user_id))
        if cursor.fetchone():
            return jsonify({'success': False, 'message': 'Username already taken'})

        # Update profile
        cursor.execute("""
            UPDATE users 
            SET name = %s, username = %s, gender = %s 
            WHERE id = %s
        """, (data['name'], data['username'], data['gender'], user_id))
        
        conn.commit()
        
        # Update session data
        session['user']['name'] = data['name']
        session['user']['username'] = data['username']
        session['user']['gender'] = data['gender']
        
        return jsonify({'success': True})
        
    except mysql.connector.Error as err:
        return jsonify({'success': False, 'message': str(err)})
    finally:
        cursor.close()
        conn.close()

@app.route('/change_password', methods=['POST'])
def change_password():
    if 'user' not in session:
        return jsonify({'success': False, 'message': 'Not logged in'}), 401

    data = request.get_json()
    user_id = session['user']['id']

    if not validate_password(data['new_password']):
        return jsonify({
            'success': False, 
            'message': 'Password must be 8+ chars with uppercase, number, and special char'
        })

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Verify current password - changed to use numeric index since it's a tuple
        cursor.execute("SELECT password FROM users WHERE id = %s", (user_id,))
        result = cursor.fetchone()
        
        if not result or result[0] != data['current_password']:
            return jsonify({'success': False, 'message': 'Current password is incorrect'})
        
        # Update password
        cursor.execute("UPDATE users SET password = %s WHERE id = %s", 
                      (data['new_password'], user_id))
        conn.commit()
        
        return jsonify({'success': True})
        
    except mysql.connector.Error as err:
        return jsonify({'success': False, 'message': str(err)})
    finally:
        cursor.close()
        conn.close()

@app.route('/delete_account', methods=['POST'])
def delete_account():
    if 'user' not in session:
        return jsonify({'success': False, 'message': 'Not logged in'}), 401

    user_id = session['user']['id']

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Delete user
        cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
        conn.commit()
        
        return jsonify({'success': True})
        
    except mysql.connector.Error as err:
        return jsonify({'success': False, 'message': str(err)})
    finally:
        cursor.close()
        conn.close()


@app.route('/get_prediction')
def get_prediction():
    return jsonify(latest_prediction)

@app.route('/get_suggestions')
def get_suggestions():
    return jsonify({
        'suggestions': suggestions_list,
        'current_index': current_index
    })

@app.route('/get_sequence')
def get_sequence():
    return jsonify({'sequence': sequence})

@app.route('/select_suggestion', methods=['POST'])
def select_suggestion():
    if 'user' not in session:
        return jsonify({'success': False, 'message': 'Not logged in'}), 401

    data = request.get_json()
    index = data.get('index', 0)
    
    global sequence, suggestions_list, current_index
    
    if suggestions_list and 0 <= index < len(suggestions_list):
        sequence += suggestions_list[index] + " "
        current_index = index  # Update the current index to match selection
        return jsonify({'success': True})
    
    return jsonify({'success': False, 'message': 'Invalid selection'})

@app.route('/handle_gesture', methods=['POST'])
def handle_gesture():
    if 'user' not in session:
        return jsonify({'success': False, 'message': 'Not logged in'}), 401

    data = request.get_json()
    gesture = data.get('gesture', '')
    
    global sequence, suggestions_list, current_index, last_gesture_time
    
    # Process the gesture immediately (no delay for button presses)
    if gesture == 'del':
        sequence = sequence[:-1]
        suggestions_list = []
    elif gesture == 'space':
        sequence += " "
    
    last_gesture_time = time.time()
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True)