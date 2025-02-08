from flask import Flask
from flask_restx import Api
from user_service_v1.app.routes import api as user_api
from pymongo import MongoClient

def create_app():
    app = Flask(__name__)
    app.config.from_object('user_service_v1.app.config.Config')
    api = Api(app)
    api.add_namespace(user_api, path='/users')
    
    # Initialize MongoDB client
    mongo_client = MongoClient(app.config['MONGO_URI'])
    app.mongo_client = mongo_client
    app.db = mongo_client[app.config['DATABASE_NAME']]
    app.users_collection = app.db['users']
    
    return app