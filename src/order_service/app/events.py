"""_summary_
Consumes user update events from a RabbitMQ queue and updates the corresponding 
user orders in the database.

Author:
    @TheBarzani
"""

import os
import json
from typing import Any, Dict, List, Optional
from flask import current_app
from dotenv import load_dotenv
from shared.config.rabbitmq_config import create_channel

load_dotenv()
QUEUE_NAME = os.getenv('RABBITMQ_QUEUE_NAME')

def consume_user_update_events() -> None:
    """
    Consumes user update events from a RabbitMQ queue and updates the corresponding 
    orders in the database.This function sets up a RabbitMQ consumer that listens 
    for user update events. When an event is received, it extracts the user ID, 
    emails, and delivery address from the event and updates the corresponding orders
    in the database with the new information.
    The function performs the following steps:
    1. Retrieves the RabbitMQ queue name from the application configuration.
    2. Creates a channel and connection to the RabbitMQ server.
    3. Defines a callback function to handle incoming messages.
        - Parses the event data from the message body.
        - Extracts the user ID, emails, and delivery address from the event.
        - Finds all orders associated with the user ID in the database.
        - Updates the orders with the new emails and delivery address if provided.
        - Acknowledges the message to remove it from the queue.
    4. Starts consuming messages from the queue using the defined callback function.
    Note:
        This function assumes that the application context is available and that 
        the `current_app` object provides access to the application configuration 
        and the orders collection in the database.
    Raises:
        Any exceptions raised during the processing of messages will be propagated.
    """

    channel, connection = create_channel(QUEUE_NAME)

    def callback(ch: Any, method: Any, properties: Any, body: bytes) -> None:
        event = json.loads(body)

        # Extract the data
        user_id: str = event['userId']
        emails: Optional[List[str]] = event.get('userEmails')
        delivery_address: Optional[str] = event.get('deliveryAddress')

        orders_collection = current_app.orders_collection
        old_orders: List[Dict[str, Any]] = list(orders_collection.find({'userId': user_id}))

        update_fields: Dict[str, Any] = {}
        if emails:
            update_fields['userEmails'] = emails
        if delivery_address:
            update_fields['deliveryAddress'] = delivery_address

        for order in old_orders:
            orders_collection.update_one({'orderId': order["orderId"]}, {'$set': update_fields})

        ch.basic_ack(delivery_tag=method.delivery_tag)

    channel.basic_consume(queue=QUEUE_NAME, on_message_callback=callback, auto_ack=False)
    channel.start_consuming()
