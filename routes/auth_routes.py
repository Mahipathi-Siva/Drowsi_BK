from flask import Blueprint, request, jsonify
import random
import bcrypt
import jwt
import datetime
import time
import os
from database import users_collection, blacklisted_tokens_collection
from utils.email_service import send_otp_email
from functools import wraps
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

auth = Blueprint("auth", __name__)

SECRET_KEY = os.getenv("SECRET_KEY")

# Rate Limiting to Prevent Brute-Force Attacks
limiter = Limiter(
    get_remote_address,
    default_limits=["50 per minute"]
)

# Helper Functions
def hash_password(password):
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))

def is_token_blacklisted(token):
    """ Check if the token is blacklisted """
    return blacklisted_tokens_collection.find_one({"token": token}) is not None

# JWT Authentication Middleware
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("Authorization")
        data = request.json  # Get email from request body

        if not token:
            return jsonify({"message": "Token is missing!"}), 403

        if blacklisted_tokens_collection.find_one({"token": token}):
            return jsonify({"message": "Please log in again."}), 403

        # Remove 'Bearer ' prefix if present
        if token.startswith("Bearer "):
            token = token.split(" ")[1]

        try:
            decoded_token = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            
            # Validate email from the request body against JWT email
            if "email" not in data or data["email"] != decoded_token["email"]:
                return jsonify({"message": "Email mismatch or missing"}), 403

            current_user = users_collection.find_one({"email": decoded_token["email"]})
            if not current_user:
                return jsonify({"message": "Invalid Token"}), 403

        except jwt.ExpiredSignatureError:
            return jsonify({"message": "Token has expired. Please login again."}), 401
        except jwt.InvalidTokenError:
            return jsonify({"message": "Invalid Token"}), 403

        return f(current_user, *args, **kwargs)

    return decorated


@auth.route("/signup", methods=["POST"])
def signup():
    data = request.json
    required_fields = ["name", "email", "phone_number", "license_number", "vehicle_number", "password", "confirm_password"]

    if not all(field in data for field in required_fields):
        return jsonify({"message": "Missing required fields"}), 400

    if data["password"] != data["confirm_password"]:
        return jsonify({"message": "Passwords do not match"}), 400

    if users_collection.find_one({"email": data["email"]}):
        return jsonify({"message": "Email already registered"}), 400

    user = {
        "name": data["name"],
        "email": data["email"],
        "phone_number": data["phone_number"],
        "license_number": data["license_number"],
        "vehicle_number": data["vehicle_number"],
        "password": hash_password(data["password"]),
        "day": [time.strftime("%Y-%m-%d")],
        "count": [0],
        "jwt_token": None,
        "otp": None
    }
    
    users_collection.insert_one(user)
    return jsonify({"message": "Signup successful! Please login."}), 201

@auth.route("/login", methods=["POST"])
@limiter.limit("5 per minute")  # Prevent brute-force attacks
def login():
    data = request.json
    user = users_collection.find_one({"email": data.get("email")})

    if not user or not check_password(data["password"], user["password"]):
        return jsonify({"message": "Invalid email or password"}), 401

    otp = random.randint(1000, 9999)
    users_collection.update_one({"email": data["email"]}, {"$set": {"otp": otp}})

    if send_otp_email(data["email"], otp):
        return jsonify({"message": "OTP sent to email"}), 200
    return jsonify({"message": "Failed to send OTP"}), 500


@auth.route("/logout", methods=["POST"])
@token_required
def logout(current_user):
    token = request.headers.get("Authorization")

    # Add token to blacklist
    blacklisted_tokens_collection.insert_one({"token": token, "created_at": datetime.datetime.utcnow()})

    return jsonify({"message": "Logged out successfully"}), 200


@auth.route("/otp_verification", methods=["POST"])
def otp_verification():
    data = request.json
    user = users_collection.find_one({"email": data.get("email")})

    if user and str(user.get("otp")) == str(data.get("otp")):
        users_collection.update_one({"email": data["email"]}, {"$set": {"otp": None}})
        
        # Generate JWT Token
        token = jwt.encode({
            "email": data["email"],
            "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=6)
        }, SECRET_KEY, algorithm="HS256")
        
        # Store token in MongoDB
        users_collection.update_one({"email": data["email"]}, {"$set": {"jwt_token": token}})

        return jsonify({"message": "Login successful", "token": token}), 200

    return jsonify({"message": "Invalid OTP"}), 400