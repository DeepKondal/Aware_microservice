"""_summary_
This module initializes the Flask application and sets up the necessary configurations,
including the Flask-RESTx API and MongoDB client.

Author:
    @TheBarzani
"""
from typing import Any
from flask import Flask
from flask_restx import Api
from pymongo import MongoClient
from user_service_v2.app.routes import api as user_api

def create_app() -> Flask:
    """
    Create and configure the Flask application.
    This function initializes the Flask application, configures it using the 
    settings from 'user_service_v2.app.config.Config', sets up the API namespace 
    for user-related endpoints, and initializes the MongoDB client.
    Returns:
        Flask: The configured Flask application instance.
    """

    app: Flask = Flask(__name__)
    app.config.from_object('user_service_v2.app.config.Config')
    api: Api = Api(app)
    api.add_namespace(user_api, path='/users')

    # Initialize MongoDB client
    mongo_client: MongoClient = MongoClient(app.config['MONGO_URI'])
    app.mongo_client = mongo_client
    app.db = mongo_client[app.config['DATABASE_NAME']]
    app.users_collection = app.db['users']

    return app
