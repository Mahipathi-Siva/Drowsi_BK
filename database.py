from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

# Check if MONGO_URI is set
if not MONGO_URI:
    raise ValueError("MONGO_URI environment variable is not set")

client = MongoClient(MONGO_URI)
db = client["Drowsi"]

# Collections
users_collection = db["Drowsi"]
blacklisted_tokens_collection = db["blacklisted_tokens"]
# Test connection
try:
    # The ismaster command is cheap and does not require auth.
    client.admin.command('ismaster')
    print("MongoDB connection successful")
except Exception as e:
    print(f"MongoDB connection error: {e}")
    raise