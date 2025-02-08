"""_summary_
This module defines the data models for the order service using Flask-RESTx.

Models:
    Item:
        - itemId (str): The unique identifier for an item.
        - quantity (int): Quantity of the item ordered. Must be at least 1.
        - price (float): Price of the item. Must be non-negative.
    DeliveryAddress:
        - street (str): Street address.
        - city (str): City.
        - state (str): State.
        - postalCode (str): Postal code.
        - country (str): Country.
    Order:
        - orderId (str): The unique identifier for an order.
        - userId (str): The unique identifier for an order.
        - items (list[Item]): List of items in the order.
        - userEmails (list[str]): A list of email addresses associated with the order.
        - deliveryAddress (DeliveryAddress): The delivery address of the user.
        - orderStatus (str): Current status of the order. Can be 'under process', 
          'shipping', or 'delivered'.
        - createdAt (datetime): Timestamp of when the order was created.
        - updatedAt (datetime): Timestamp of when the order was last updated.
Author:
    @TheBarzani
"""

from flask_restx import fields, Namespace

api = Namespace('orders', description='Order related operations')

item_model = api.model('Item', {
    'itemId': fields.String(required=True, description='The unique identifier for an item'),
    'quantity': fields.Integer(required=True, description='Quantity of the item ordered', min=1),
    'price': fields.Float(required=True, description='Price of the item', min=0)
})

delivery_address_model = api.model('DeliveryAddress', {
    'street': fields.String(required=True, description='Street address'),
    'city': fields.String(required=True, description='City'),
    'state': fields.String(required=True, description='State'),
    'postalCode': fields.String(required=True, description='Postal code'),
    'country': fields.String(required=True, description='Country')
})

order_model = api.model('Order', {
    'orderId': fields.String(required=True, description='The unique identifier for an order'),
    'userId': fields.String(required=True, description='The unique identifier for a user'),
    'items': fields.List(fields.Nested(item_model), required=True, description='List of '+
                         'items in the order'),
    'userEmails': fields.List(fields.String, required=True, description='A list of email '+
                              'addresses associated with the order'),
    'deliveryAddress': fields.Nested(delivery_address_model, required=True, description=
                                     'The delivery address of the user'),
    'orderStatus': fields.String(required=True, description='Current status of the order', 
                                 enum=['under process', 'shipping', 'delivered']),
    'createdAt': fields.DateTime(description='Timestamp of when the order was created.'),
    'updatedAt': fields.DateTime(description='Timestamp of when the order was last updated.')
})
