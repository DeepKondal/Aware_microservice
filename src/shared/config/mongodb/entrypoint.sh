#!/bin/bash
source .env
echo "Waiting for MongoDB to be ready..."
# Wait for MongoDB to be ready using Python
python3 << END
import time
import os
import sys

try:
    from pymongo import MongoClient
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Explicitly load .env file
dotenv_path = '/app/.env'
load_dotenv(dotenv_path)

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    print("MONGO_URI not found in .env file")
    sys.exit(1)

MAX_ATTEMPTS = 30
DELAY = 2

for attempt in range(MAX_ATTEMPTS):
    try:
        client = MongoClient(MONGO_URI)
        client.admin.command('ping')
        print("MongoDB is ready!")
        break
    except Exception as e:
        print(f"MongoDB not ready yet. Attempt {attempt + 1}/{MAX_ATTEMPTS}")
        print(f"Connection error: {e}")
        time.sleep(DELAY)
else:
    print("Failed to connect to MongoDB after multiple attempts.")
    sys.exit(1)
END

# Run setup script
python setup_mongodb.py

# Run seed database script
python seed_database.py

echo "MongoDB setup and seeding complete."