"""_summary_
This module initializes and configures the Flask application for the order service.
It sets up the Flask app, configures the API namespace, initializes the MongoDB 
client, and starts a background thread to consume user update events.

Functions:
    start_event_consumer(app: Flask): Starts the event consumer within the Flask 
                                      app context.
    create_app(): Creates and configures the Flask application, initializes 
                  MongoDB, and starts the event consumer thread.
Athor:
    @TheBarzani
"""

import threading
from flask import Flask
from pymongo import MongoClient
from flask_restx import Api
from order_service.app.routes import api as order_api
from order_service.app.events import consume_user_update_events

def start_event_consumer(app: Flask) -> None:
    """
    Starts the event consumer for the given Flask application.
    This function initializes the event consumer within the application context
    and begins consuming user update events.
    Args:
        app (Flask): The Flask application instance.
    Returns:
        None
    """

    # print("Starting event consumer...")
    with app.app_context():
        consume_user_update_events()

def create_app() -> Flask:
    """
    Create and configure the Flask application.
    This function initializes the Flask application, configures it using the 
    settings from 'config.py', sets up the API namespace for order-related 
    endpoints, and initializes the MongoDB client. It also starts the event 
    consumer in a separate thread.
    Returns:
        Flask: The configured Flask application instance.
    """

    app = Flask(__name__)
    app.config.from_object('order_service.app.config.Config')
    api = Api(app)
    api.add_namespace(order_api, path='/orders')

    # Initialize MongoDB client
    # print ("Connecting to MongoDB... ", app.config['MONGO_URI'])
    mongo_client = MongoClient(app.config['MONGO_URI'])
    app.mongo_client = mongo_client
    app.db = mongo_client[app.config['DATABASE_NAME']]
    app.orders_collection = app.db['orders']

    # Start the event consumer in a separate thread
    event_consumer_thread = threading.Thread(target=start_event_consumer, args=(app,), daemon=True)
    event_consumer_thread.start()
    return app
