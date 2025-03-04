import os
from flask import Flask, jsonify
import secrets
from flask_cors import CORS
from dotenv import load_dotenv
from flask_socketio import SocketIO
from routes.auth_routes import auth, limiter, token_required
from routes.user_routes import user
from database import users_collection, blacklisted_tokens_collection, db
from flask import Blueprint, request, Response
import cv2
import mediapipe as mp
import numpy as np
import time
from scipy.spatial import distance
import face_recognition
from ultralytics import YOLO
import pygame
from flask import Flask, render_template, Response, request, redirect, url_for, session, flash, jsonify
from flask_session import Session
from pymongo import MongoClient
import threading
from flask_socketio import SocketIO, emit
from io import BytesIO
from PIL import Image
import bcrypt
import random
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import re
from scipy.signal import medfilt
import base64  # For decoding/encoding images
import queue
import base64

# A queue to store the latest camera frame received from the frontend
frame_queue = queue.Queue(maxsize=1)


# Load environment variables first
load_dotenv()

# Create Flask app
app = Flask(__name__)
CORS(app)

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*")

video_streaming = True

pygame.mixer.init()
alert_sound = pygame.mixer.Sound("music2.mp3")

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

@socketio.on('camera_frame')
def handle_camera_frame(data):
    global frame_queue
    # Expecting data to have a key 'image' containing a Base64 encoded JPEG
    img_data = data.get('image')
    if not img_data:
        return
    # Remove header if it exists, e.g. "data:image/jpeg;base64,"
    if "," in img_data:
        header, encoded = img_data.split(",", 1)
    else:
        encoded = img_data
    try:
        decoded_data = base64.b64decode(encoded)
        np_arr = np.frombuffer(decoded_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print("Error decoding image:", e)
        return

    # Replace the old frame in the queue with the new one
    if frame_queue.full():
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            pass
    frame_queue.put(frame)


# Load YOLO model
model = YOLO('best.pt')

# Utility functions for ML detection
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[3], mouth[9])
    B = distance.euclidean(mouth[2], mouth[10])
    C = distance.euclidean(mouth[4], mouth[8])
    D = distance.euclidean(mouth[0], mouth[6])
    mar = (A + B + C) / (2.0 * D)
    return mar

# -------------------------------
# Existing streaming detection (for testing)
def combined_detection(email):
    EYE_AR_THRESH = 0.25
    MOUTH_AR_THRESH = 0.42
    score = 0
    count = 0
    ear_history = []
    EAR_WINDOW = 10
    blinks = 0
    blink_start = None
    BLINK_DURATION = 3
    yawn_count = 0
    head_pose_timer = None
    ALERT_DELAY = 5  # seconds
    ALERT_DELAY1 = 8  # seconds

    while video_streaming:
        if not video_streaming:
            break
        # Wait for a new frame from the queue
        if frame_queue.empty():
            time.sleep(0.01)
            continue
        try:
            # Get the latest frame from the queue
            image = frame_queue.get()
            if image is None:
                continue

            # Flip the image horizontally and convert to RGB
            image_rgb = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

            # To improve performance
            image_rgb.flags.writeable = False
            
            # Get face mesh results
            results = face_mesh.process(image_rgb)

            image_rgb.flags.writeable = True
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            img_h, img_w, img_c = image_bgr.shape
            face_3d = []
            face_2d = []

            # Drowsiness Detection
            rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            count += 1

            if face_locations:
                face_landmarks_list = face_recognition.face_landmarks(rgb_frame, face_locations)

                for face_landmarks in face_landmarks_list:
                    left_eye = np.array(face_landmarks['left_eye'])
                    right_eye = np.array(face_landmarks['right_eye'])
                    mouth = np.array(face_landmarks['bottom_lip'])
                    print(f"Bottom Lip Landmarks: {face_landmarks['bottom_lip']}")

                    # Calculate EAR and MAR
                    left_ear = eye_aspect_ratio(left_eye)
                    right_ear = eye_aspect_ratio(right_eye)
                    ear = (left_ear + right_ear) / 2.0
                    ear_history.append(ear)
                    if len(ear_history) > EAR_WINDOW:
                         ear_history.pop(0)
                    
                    adaptive_threshold = max(0.2, np.mean(ear_history) * 0.8)
                    eye_flag = ear < adaptive_threshold

                    #mar = mouth_aspect_ratio(mouth)

                    # Check if eyes are closed
                    #eye_flag = ear < EYE_AR_THRESH
                    # Check if mouth is open
                    #mouth_flag = mar > MOUTH_AR_THRESH



                    # Update score more responsively
                    if eye_flag :
                        score += 1
                    #elif mouth_flag:
                        #score +=1  # Increase score faster
                    else:
                        score -= 1  # Decrease score more slowly
                    if score < 0:
                        score = 0

                    if eye_flag:
                        if blink_start is None:
                            blink_start = time.time()
                    else:
                        if blink_start and time.time() - blink_start > BLINK_DURATION / 30:
                            blinks += 1
                        blink_start = None
                    cv2.putText(image_bgr, f"Blinks/min: {blinks}", (20, 80), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    
                    mar = mouth_aspect_ratio(mouth)
                    print(f"MAR: {mar}")
                    mouth_flag = mar > MOUTH_AR_THRESH

                '''if -10 < y < 10:
                        if mouth_flag:
                            yawn_count += 1
                        #print(f"Yawning detected! Count: {yawn_count}")
                        else:
                            yawn_count -= 1
                    if yawn_count < 0:
                        yawn_count = 0
                    current_time = time.time() '''
                    
            

               
            y = 0  
            x = 0
            z = 0
            # Check Head Pose
            head_pose_alert = False
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    for idx, lm in enumerate(face_landmarks.landmark):
                        if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                            if idx == 1:
                                nose_2d = (lm.x * img_w, lm.y * img_h)
                                nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                            x, y = int(lm.x * img_w), int(lm.y * img_h)

                            # Get the 2D Coordinates
                            face_2d.append([x, y])

                            # Get the 3D Coordinates
                            face_3d.append([x, y, lm.z])

                    # Convert it to the NumPy array
                    face_2d = np.array(face_2d, dtype=np.float64)
                    face_3d = np.array(face_3d, dtype=np.float64)

                    # Camera matrix
                    focal_length = 1 * img_w
                    cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                            [0, focal_length, img_w / 2],
                                            [0, 0, 1]])

                    dist_matrix = np.zeros((4, 1), dtype=np.float64)

                    # Solve PnP
                    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                    # Get rotational matrix
                    rmat, jac = cv2.Rodrigues(rot_vec)

                    # Get angles
                    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                    # Get the y rotation degree
                    x = angles[0] * 360
                    y = angles[1] * 360
                    z = angles[2] * 360

                    # See where the user's head is tilting
                    if y < -10:
                        text = "Looking Left"
                        if head_pose_timer is None:
                            head_pose_timer = time.time()
                        elif time.time() - head_pose_timer > ALERT_DELAY1:
                            head_pose_alert = True
                            cv2.putText(image_bgr, "Distracted", (image_bgr.shape[1] - 200, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    elif y > 10:
                        text = "Looking Right"
                        if head_pose_timer is None:
                            head_pose_timer = time.time()
                        elif time.time() - head_pose_timer > ALERT_DELAY1:
                            head_pose_alert = True
                            cv2.putText(image_bgr, "Distracted", (image_bgr.shape[1] - 200, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    elif x < -10:
                        text = "Looking Down"
                        if head_pose_timer is None:
                            head_pose_timer = time.time()
                        elif time.time() - head_pose_timer > ALERT_DELAY:
                            head_pose_alert = True
                            cv2.putText(image_bgr, "focus on the steering", (image_bgr.shape[1] - 400, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    elif x > 10:
                        text = "Looking Up"
                        if head_pose_timer is None:
                            head_pose_timer = time.time()
                        elif time.time() - head_pose_timer > ALERT_DELAY:
                            head_pose_alert = True
                            cv2.putText(image_bgr, "focus on the steering", (image_bgr.shape[1] - 400, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    else:
                        text = "Forward"
                        head_pose_timer = None

                    if -10 < y < 10:
                        if mouth_flag:
                             yawn_count += 1
                        else:
                             yawn_count -= 1
                    if yawn_count < 0:
                         yawn_count = 0
                    

                    # Display the nose direction
                    nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
                    p1 = (int(nose_2d[0]), int(nose_2d[1]))
                    p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))
                    cv2.line(image_bgr, p1, p2, (255, 0, 0), 3)
                    cv2.putText(image_bgr, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

            # Display drowsiness score and warning
            cv2.putText(image_bgr, f"Score: {score}", (10, image_bgr.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            

            if score >= 6:
                cv2.putText(image_bgr, "DROWSY!", (image_bgr.shape[1] - 200, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                head_pose_alert = True

            (text_width, text_height), _ = cv2.getTextSize(f"Score: {score}", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

            cv2.putText(image_bgr, f"Yawn: {yawn_count}", (10, image_bgr.shape[0] - 40),  # Move up
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            if yawn_count >= 6 :
                cv2.putText(image_bgr, "Continuous Yawning", (image_bgr.shape[1] - 300, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                head_pose_alert = True



            # Check for cell phone detection
            results = model(image)
            names = model.names
            for r in results:
                for c in r.boxes.cls:
                    if(names[int(c)] == "mobile"):
                        print("Cell phone detected!")
                        head_pose_alert = True
                        cv2.putText(image_bgr, "Mobile Phone  detected", (image_bgr.shape[1] - 400, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
           # Inside your combined_detection function

            if head_pose_alert:
                today_date = time.strftime("%Y-%m-%d")  # Get today's date in YYYY-MM-DD format

                # Check if user exists
                user = users_collection.find_one({"email": email})
                print(f"Today Date: {today_date}, Email: {email}, User: {user}")
                
                if user:  # Ensure user exists
                    # If today_date is already present in the user's data
                    if today_date in user["day"]:
                        index = user["day"].index(today_date)
                        # Update the count at the specific index of both arrays
                        users_collection.update_one(
                            {"email": email},
                            {"$inc": {f"count.{index}": 1}}  # Increment count at the specific index
                        )
                    else:
                        users_collection.update_one(
                            {"email": email},
                            {"$push": {"day": today_date}, "$addToSet": {"count": 0}}
                        )


                   

            if head_pose_alert:
                socketio.emit('alert', {'message': 'Drowsiness detected!'}, namespace='/alert')
                alert_sound.play()
                
                

            ret, buffer = cv2.imencode('.jpg', image_bgr)
            frame = buffer.tobytes()

            yield(b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            print(f"Error processing frame: {e}")
            break

# Initialize database connection
try:
    from database import users_collection, blacklisted_tokens_collection, db
    # Create alerts collection in the same database
    user_collection = db["Drowsi"]
    print("Database imported successfully")
except Exception as e:
    print(f"Error importing database: {e}")
    raise

# Import routes after database is initialized
try:
    from routes.auth_routes import auth, limiter, token_required
    from routes.user_routes import user
    print("Routes imported successfully")
except Exception as e:
    print(f"Error importing routes: {e}")
    raise

# Import the drowsiness detection module


# Register blueprints
app.register_blueprint(auth, url_prefix="/auth")
app.register_blueprint(user, url_prefix="/user")

# Initialize rate limiter
limiter.init_app(app)


# Diagnostic endpoint
@app.route("/debug", methods=["GET"])
def debug():
    try:
        # Test database connection
        user_count = users_collection.count_documents({})
        email_configured = bool(os.getenv("EMAIL_USER") and os.getenv("EMAIL_PASS"))
        
        return jsonify({
            "status": "ok",
            "database_connected": True,
            "user_count": user_count,
            "email_configured": email_configured,
            "mongo_uri_configured": bool(os.getenv("MONGO_URI")),
            "secret_key_configured": bool(os.getenv("SECRET_KEY")),
            "detection_module": "loaded"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# Root endpoint
@app.route("/", methods=["GET"])
def home():
    return "Drowsi API is running!"

@app.route('/index')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print("Client connected!")

@app.route('/video')
def video():
    global video_streaming
    # Reset the flag so that streaming starts fresh
    video_streaming = True
    email = "mahipathi.31@gmail.com"
    return Response(combined_detection(email), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_video')
def stop_video():
    global video_streaming
    video_streaming = False
    return jsonify({"message": "Video streaming stopped."})


if __name__ == "__main__":
    print("Starting Drowsi application...")
    # Run with socketio instead of regular app.run
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)