"""_summary_
This script seeds a MongoDB database with sample data for users and orders collections.
It connects to the MongoDB instance using credentials from a .env file, drops existing 
collections, and inserts new sample data.

Functions:
    seed_users() -> List[Dict[str, Any]]:
        Seeds the users collection with sample user data.
        Returns a list of seeded user documents.

    seed_orders(users: List[Dict[str, Any]]) -> None:
        Seeds the orders collection with sample order data.
        Takes a list of user documents to associate orders with users.
Author:
    @TheBarzani        
"""

import random
import os
from typing import List, Dict, Any
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Retrieve MongoDB credentials from .env
MONGO_URI = os.getenv("MONGO_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME")

if not MONGO_URI or not DATABASE_NAME:
    raise ValueError("MongoDB URI or database name is not set in the .env file.")

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]

# Sample data for seeding
CITIES = ["Montreal", "Toronto", "Vancouver", "Calgary", "Ottawa"]
COUNTRIES = ["Canada", "USA", "UK", "Germany", "France"]
STATUSES = ["under process", "shipping", "delivered"]

# Seed Users Collection
def seed_users() -> List[Dict[str, Any]]:
    """
    Seed the MongoDB database with user data.
    
    The generated user data is then inserted into the 'users' collection in the MongoDB 
    database.
    
    Returns:
        List[Dict[str, Any]]: A list of dictionaries representing the seeded users.
    """

    print("Seeding users...")
    users: List[Dict[str, Any]] = []
    for i in range(5):  # Create 5 users
        user: Dict[str, Any] = {
            "userId": f"u{i+1}",
            "firstName": f"firstname-{i+1}",
            "lastName": f"lastname-{i+1}",
            "emails": [f"user{i+1}@example.com"],
            "deliveryAddress": {
                "street": f"{random.randint(100, 999)} Example St",
                "city": random.choice(CITIES),
                "state": "State-" + str(random.randint(1, 10)),
                "postalCode": f"{random.randint(10000, 99999)}",
                "country": random.choice(COUNTRIES)
            },
            "phoneNumber": f"{random.randint(1000000000, 9999999999)}",
            "createdAt": datetime.utcnow(),
            "updatedAt": datetime.utcnow()
        }
        users.append(user)
    db.users.insert_many(users)
    print(f"Seeded {len(users)} users.")
    return users

# Seed Orders Collection
def seed_orders(users: List[Dict[str, Any]]) -> None:
    """
    Seed the orders collection in MongoDB with sample data.
    This function generates a list of 15 orders, each containing 1 to 3 items,
    and inserts them into the MongoDB orders collection. Each order is associated
    with a randomly selected user from the provided list of users.
    
    Args:
        users (List[Dict[str, Any]]): A list of user dictionaries. Each dictionary
                                      should contain the keys "emails" and "deliveryAddress".
    Returns:
        None
    Raises:
        pymongo.errors.PyMongoError: If an error occurs while inserting the orders into MongoDB.
    Example:
        users = [
                "emails": ["user1@example.com"],
                "deliveryAddress": "123 Main St, Anytown, USA"
            },
                "emails": ["user2@example.com"],
                "deliveryAddress": "456 Elm St, Othertown, USA"
        ]
        seed_orders(users)
    """

    print("Seeding orders...")
    orders: List[Dict[str, Any]] = []
    for i in range(15):  # Create 15 orders
        user: Dict[str, Any] = random.choice(users)
        order: Dict[str, Any] = {
            "orderId": f"o{i+1}",
            "userId": user["userId"],
            "items": [
                {
                    "itemId": f"item{j+1}",
                    "name": f"Item {j+1}",
                    "quantity": random.randint(1, 5),
                    "price": round(random.uniform(10.0, 200.0), 2)
                }
                for j in range(random.randint(1, 3))  # 1–3 items per order
            ],
            "userEmails": user["emails"],
            "deliveryAddress": user["deliveryAddress"],
            "orderStatus": random.choice(STATUSES),
            "createdAt": datetime.utcnow(),
            "updatedAt": datetime.utcnow()
        }
        orders.append(order)
    db.orders.insert_many(orders)
    print(f"Seeded {len(orders)} orders.")

# Main function
def main() -> None:
    """
    This function seeds the MongoDB database with sample user and order data.
    """
    print("Dropping existing collections...")

    print("Seeding database...")
    users: List[Dict[str, Any]] = seed_users()
    seed_orders(users)
    print("Database seeding complete.")
    # Retrieve and print one user
    user: Dict[str, Any] = db.users.find_one()
    print("Sample User:")
    print(user)

    # Retrieve and print one order
    order: Dict[str, Any] = db.orders.find_one()
    print("Sample Order:")
    print(order)

if __name__ == "__main__":
    main()
