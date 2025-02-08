import pika
import os
from dotenv import load_dotenv

load_dotenv()

RABBITMQ_HOST = os.getenv('RABBITMQ_HOST')
RABBITMQ_PORT = int(os.getenv('RABBITMQ_PORT'))
RABBITMQ_USER = os.getenv('RABBITMQ_USER', 'admin')
RABBITMQ_PASSWORD = os.getenv('RABBITMQ_PASSWORD', 'admin')
EXCHANGE_NAME = 'my_exchange'
QUEUE_NAME = 'my_queue'
ROUTING_KEY = 'my_routing_key'

def get_connection():
    credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASSWORD)
    return pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST, port=RABBITMQ_PORT, credentials=credentials))

def setup_rabbitmq():
    connection = get_connection()
    channel = connection.channel()

    # Declare an exchange
    channel.exchange_declare(exchange=EXCHANGE_NAME, exchange_type='direct', durable=True)

    # Declare a queue
    channel.queue_declare(queue=QUEUE_NAME, durable=True)

    # Bind the queue to the exchange with a routing key
    channel.queue_bind(exchange=EXCHANGE_NAME, queue=QUEUE_NAME, routing_key=ROUTING_KEY)

    return channel, connection

def publish_message(message):
    channel, connection = setup_rabbitmq()
    channel.basic_publish(exchange=EXCHANGE_NAME, routing_key=ROUTING_KEY, body=message)
    connection.close()

def consume_messages():
    channel, connection = setup_rabbitmq()

    def callback(ch, method, properties, body):
        print(f"Received message: {body}")

    channel.basic_consume(queue=QUEUE_NAME, on_message_callback=callback, auto_ack=True)
    print("Waiting for messages...")
    channel.start_consuming()

# Example usage
if __name__ == "__main__":
    publish_message("Hello, RabbitMQ!")
    consume_messages()