'''
Version 0.0.3 - Enhanced Neo4j tracking, detailed RabbitMQ messages, and better error handling.
'''

from fastapi import FastAPI, HTTPException, Request
import httpx
import json
import os
import asyncio
import logging
import datetime
import pika  # RabbitMQ integration
from adversarial.adversarial_model import evaluate_adversarial
from neo4j_client import ProvenanceModel

# Load environment variables
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
RABBITMQ_QUEUE_NAME = os.getenv("RABBITMQ_QUEUE_NAME", "pipeline_tasks")

# Initialize FastAPI app
app = FastAPI(title="Coordination Center")

# Initialize Neo4j Provenance Model
provenance = ProvenanceModel(uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD)

# 🔹 RabbitMQ Connection Setup
def send_rabbitmq_message(event, run_id=None, dataset_id=None, error=None):
    """Send a structured message to RabbitMQ."""
    try:
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))
        channel = connection.channel()
        channel.queue_declare(queue=RABBITMQ_QUEUE_NAME, durable=True)
        
        message = {"event": event, "timestamp": str(datetime.datetime.now())}
        if run_id:
            message["run_id"] = run_id
        if dataset_id:
            message["dataset_id"] = dataset_id
        if error:
            message["error"] = str(error)

        channel.basic_publish(exchange='', routing_key=RABBITMQ_QUEUE_NAME, body=json.dumps(message))
        connection.close()
        print(f"✅ Sent RabbitMQ Message: {message}")
    except Exception as e:
        logging.error(f"❌ Failed to send RabbitMQ message: {e}")

# 🔹 Asynchronous HTTP Request Handler
async def async_http_post(url, json_data=None):
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(url, json=json_data)
        if response.status_code != 200:
            logging.error(f"❌ Error in POST to {url}: {response.status_code} - {response.text}")
            raise HTTPException(status_code=response.status_code, detail=response.text)
        return response.json()

# 🔹 Upload Dataset Processing
async def process_upload_config(upload_config, run_id):
    for dataset_id, dataset_info in upload_config['datasets'].items():
        local_video_dir = dataset_info.get('local_video_dir')
        if not local_video_dir or not os.path.exists(local_video_dir):
            raise HTTPException(status_code=400, detail=f"Video directory {local_video_dir} not found.")
        
        url = upload_config['server_url'] + "/process-kinetics-dataset"
        json_data = {"video_dir": local_video_dir, "num_frames": 8}

        try:
            await async_http_post(url, json_data=json_data)
            
            # 🔄 Log dataset creation in Neo4j
            provenance.create_dataset(dataset_id, "Kinetics-400", local_video_dir)

            # 🔄 Track Upload Step in Neo4j
            provenance.create_processing_step("Upload Data", "upload", str(upload_config))
            provenance.link_pipeline_step(run_id, "Upload Data")
            provenance.link_dataset_to_processing(dataset_id, "Upload Data")

            # 🚀 Send Upload Success Message to RabbitMQ
            send_rabbitmq_message("upload_completed", run_id, dataset_id)
            print(f"✅ Processed video dataset {dataset_id} successfully.")

        except Exception as e:
            send_rabbitmq_message("upload_failed", run_id, dataset_id, str(e))
            logging.error(f"❌ Upload failed: {e}")

# 🔹 Pipeline Execution
async def process_pipeline_from_config(config):
    run_id = f"run_{datetime.datetime.now().isoformat()}"

    try:
        # 🔄 Create pipeline run node in Neo4j
        provenance.create_pipeline_run(run_id, datetime.datetime.now().isoformat(), "Running")

        # 🚀 Notify RabbitMQ
        send_rabbitmq_message("pipeline_started", run_id)

        # 🟢 Upload Processing
        await process_upload_config(config['upload_config'], run_id)

        # 🔄 Update Pipeline Completion in Neo4j
        provenance.update_pipeline_status(run_id, "Completed")

        # 🚀 Notify RabbitMQ that Pipeline is Completed
        send_rabbitmq_message("pipeline_completed", run_id)

    except Exception as e:
        provenance.update_pipeline_status(run_id, "Failed")
        send_rabbitmq_message("pipeline_failed", run_id, error=str(e))
        logging.error(f"❌ Pipeline processing failed: {e}")

# 🔹 Run Pipeline Endpoint
@app.post("/run_pipeline/")
async def run_pipeline(request: Request):
    """Trigger pipeline execution from API."""
    config = await request.json()
    run_id = f"run_{datetime.datetime.now().isoformat()}"
    
    # 🚀 Notify RabbitMQ
    send_rabbitmq_message("pipeline_started", run_id)

    try:
        await process_pipeline_from_config(config)
        return {"message": f"✅ Pipeline {run_id} executed successfully"}
    except Exception as e:
        send_rabbitmq_message("pipeline_failed", run_id, error=str(e))
        return HTTPException(status_code=500, detail=str(e))

# 🔹 Run Server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8880)
