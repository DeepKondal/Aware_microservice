import os
import shutil
from PIL import Image
import numpy as np
import random, av
import pandas as pd
import aiofiles
import asyncio
import json
import logging
import datetime
import pika  # RabbitMQ Integration
from neo4j_client import ProvenanceModel
from azure.storage.blob.aio import BlobServiceClient

class DataProcess:
    def __init__(self, base_storage_address):
        self.base_storage_address = base_storage_address 
        self.metadata = {}  # Removed `self.datasets`, using Neo4j instead
        
        # Initialize Neo4j Client
        self.provenance = ProvenanceModel()

        # RabbitMQ Setup
        self.rabbitmq_host = os.getenv("RABBITMQ_HOST", "localhost")
        self.rabbitmq_queue = os.getenv("RABBITMQ_QUEUE_NAME", "pipeline_tasks")

    # 🟢 RabbitMQ Messaging Function
    def send_rabbitmq_message(self, event, dataset_id=None, error=None):
        """Send RabbitMQ messages for dataset events."""
        try:
            connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.rabbitmq_host))
            channel = connection.channel()
            channel.queue_declare(queue=self.rabbitmq_queue, durable=True)
            
            message = {"event": event, "timestamp": str(datetime.datetime.now())}
            if dataset_id:
                message["dataset_id"] = dataset_id
            if error:
                message["error"] = str(error)

            channel.basic_publish(exchange='', routing_key=self.rabbitmq_queue, body=json.dumps(message))
            connection.close()
            print(f"✅ Sent RabbitMQ Message: {message}")
        except Exception as e:
            logging.error(f"❌ Failed to send RabbitMQ message: {e}")

    # 📌 Upload Dataset and Register in Neo4j
    async def upload_dataset(self, data_files, dataset_id, data_type):
        """Asynchronously uploads dataset & registers in Neo4j."""
        dataset_dir = os.path.join(self.base_storage_address, data_type, dataset_id)
        os.makedirs(dataset_dir, exist_ok=True)

        for file_path in data_files:
            label = os.path.basename(os.path.dirname(file_path))
            label_dir = os.path.join(dataset_dir, label)
            os.makedirs(label_dir, exist_ok=True)

            file_extension = os.path.splitext(file_path)[1]
            dest_file_name = os.path.splitext(os.path.basename(file_path))[0] + file_extension
            dest_file_path = os.path.join(label_dir, dest_file_name)

            async with aiofiles.open(file_path, 'rb') as src, aiofiles.open(dest_file_path, 'wb') as dst:
                await dst.write(await src.read())

        print(f"✅ Dataset '{dataset_id}' uploaded.")

        # 🔄 Store dataset in Neo4j
        self.provenance.create_dataset(dataset_id, f"{data_type} Dataset", dataset_dir)

        # 🚀 Notify RabbitMQ
        self.send_rabbitmq_message("dataset_uploaded", dataset_id)

    # 📌 Delete Dataset
    def delete_dataset(self, dataset_id):
        """Deletes dataset from storage and Neo4j."""
        dataset_dir = os.path.join(self.base_storage_address, dataset_id)
        shutil.rmtree(dataset_dir, ignore_errors=True)

        # 🔄 Remove dataset from Neo4j
        self.provenance.driver.session().run("MATCH (d:Dataset {id: $dataset_id}) DETACH DELETE d", dataset_id=dataset_id)

        print(f"❌ Deleted dataset '{dataset_id}'.")
        self.send_rabbitmq_message("dataset_deleted", dataset_id)

    # 📌 Apply Image Perturbations
    async def apply_image_perturbation(self, dataset_id, perturbation_func, severity=1):
        """Applies perturbation & logs in Neo4j."""
        dataset_dir = os.path.join(self.base_storage_address, dataset_id)

        perturbed_folder_name = f"{dataset_id}_{perturbation_func.__name__}_{severity}"
        perturbed_folder_path = os.path.join(dataset_dir, "..", perturbed_folder_name)
        os.makedirs(perturbed_folder_path, exist_ok=True)

        tasks = []
        for label in os.listdir(dataset_dir):
            label_dir = os.path.join(dataset_dir, label)
            if not os.path.isdir(label_dir):
                continue

            perturbed_label_dir = os.path.join(perturbed_folder_path, label)
            os.makedirs(perturbed_label_dir, exist_ok=True)

            for file in os.listdir(label_dir):
                file_path = os.path.join(label_dir, file)
                if os.path.isfile(file_path) and file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    tasks.append(self._process_and_save_image(file_path, perturbed_label_dir, perturbation_func, severity))

        await asyncio.gather(*tasks)

        # 🔄 Log perturbation in Neo4j
        self.provenance.create_processing_step("Apply Perturbation", "perturbation", f"Applied {perturbation_func.__name__} with severity {severity}")
        self.provenance.link_dataset_to_processing(dataset_id, "Apply Perturbation")

        # 🚀 Notify RabbitMQ
        self.send_rabbitmq_message("perturbation_applied", dataset_id)

        return perturbed_folder_path

    # 📌 Process Kinetics Videos
    def process_kinetics_video(self, video_path, num_frames=8):
        """Processes Kinetics dataset videos."""
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(self.base_storage_address, "videos", video_name)
        os.makedirs(output_dir, exist_ok=True)

        container = av.open(video_path)
        frames = [frame.to_image() for frame in container.decode(video=0)]
        sampled_frames = [frames[i] for i in np.linspace(0, len(frames) - 1, num_frames, dtype=int)]

        for idx, frame in enumerate(sampled_frames):
            frame.save(os.path.join(output_dir, f"frame_{idx + 1}.jpg"))

        # 🔄 Log dataset in Neo4j
        self.provenance.create_dataset(video_name, "Kinetics Video", video_path)
        self.provenance.create_processing_step("Extract Frames", "video_processing", f"Extracted {num_frames} frames from {video_name}")
        self.provenance.link_dataset_to_processing(video_name, "Extract Frames")

        print(f"✅ Processed {video_name} - Frames saved in {output_dir}")
        return {"video_name": video_name, "frame_dir": output_dir}
