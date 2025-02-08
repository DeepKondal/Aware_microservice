"""__summary__
This module defines the data models for the user service using Flask-RESTx.

Models:
    delivery_address_model (Model): A model representing a delivery address with 
                                    the following fields:
        - street (String): Street address (required).
        - city (String): City (required).
        - state (String): State (required).
        - postalCode (String): Postal code (required).
        - country (String): Country (required).
    user_model (Model): A model representing a user with the following fields:
        - userId (String): The unique identifier for a user account (required).
        - firstName (String): First name of the user.
        - lastName (String): Last name of the user.
        - emails (List[String]): A list of email addresses associated with the user (required).
        - deliveryAddress (Nested): The delivery address of the user (required).
        - phoneNumber (String): Optional phone number for the user, 10-15 digits.
        - createdAt (DateTime): Timestamp of when the user was created.
        - updatedAt (DateTime): Timestamp of when the user was last updated.
Author:
    @TheBarzani
"""
from flask_restx import fields, Namespace

api = Namespace('users', description='User related operations')

delivery_address_model = api.model('DeliveryAddress', {
    'street': fields.String(required=True, description='Street address'),
    'city': fields.String(required=True, description='City'),
    'state': fields.String(required=True, description='State'),
    'postalCode': fields.String(required=True, description='Postal code'),
    'country': fields.String(required=True, description='Country')
})

user_model = api.model('User', {
    'userId': fields.String(required=True, description='The unique identifier for a user account'),
    'firstName': fields.String(description='First name of the user'),
    'lastName': fields.String(description='Last name of the user'),
    'emails': fields.List(fields.String, required=True, description='A list of email'+
                          ' addresses associated with the user'),
    'deliveryAddress': fields.Nested(delivery_address_model, required=True, description=
                                     'The delivery address of the user'),
    'phoneNumber': fields.String(pattern='^[0-9]{10,15}$', description='Optional phone' +
                                 ' number for the user, 10-15 digits.'),
    'createdAt': fields.DateTime(description='Timestamp of when the user was created.'),
    'updatedAt': fields.DateTime(description='Timestamp of when the user was last updated.')
})
