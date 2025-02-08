"""__summary__
This module handles the publishing of user update events to a RabbitMQ queue.

Author:
    @TheBarzani
"""

import os
import json
from flask import current_app
from dotenv import load_dotenv
from shared.config.rabbitmq_config import create_channel

load_dotenv()
QUEUE_NAME = os.getenv('RABBITMQ_QUEUE_NAME')

def publish_user_update_event(user_id: int, email: str, address: str) -> None:
    """
    Publishes an event to notify about a user update.
    Args:
        user_id (int): The ID of the user.
        email (str): The email address of the user.
        address (str): The delivery address of the user.
    Returns:
        None  
    """

    channel, connection = create_channel(QUEUE_NAME)
    event = {
        'userId': user_id,
        'userEmails': email,
        'deliveryAddress': address
    }
    channel.basic_publish(
        exchange="user_order",
        routing_key=QUEUE_NAME,
        body=json.dumps(event)
        # properties=pika.BasicProperties(
        #     delivery_mode=2,  # Make the message persistent
        # )
    )
    print(f"V2 Published event: {event}", flush=True)
    connection.close()
