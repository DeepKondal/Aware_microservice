from pydantic import BaseModel
from typing import List
import logging
import cam_resnet
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
import aiohttp
import os
import json
import datetime
import torch
import pika  # RabbitMQ Integration
from attention_extractor import AttentionExtractor
from neo4j_client import ProvenanceModel  # Neo4j Integration

# Initialize FastAPI app
app = FastAPI(title="XAI Service")

# Initialize STAA Model
staa_model = AttentionExtractor("facebook/timesformer-base-finetuned-k400", device="cpu")

# 🔗 Initialize Neo4j Provenance Tracking
provenance = ProvenanceModel()

# 🚀 RabbitMQ Setup
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
RABBITMQ_QUEUE_NAME = os.getenv("RABBITMQ_QUEUE_NAME", "pipeline_tasks")

def send_rabbitmq_message(event, dataset_id=None, video_file=None, error=None):
    """Send RabbitMQ messages for XAI processing events."""
    try:
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))
        channel = connection.channel()
        channel.queue_declare(queue=RABBITMQ_QUEUE_NAME, durable=True)

        message = {"event": event, "timestamp": str(datetime.datetime.now())}
        if dataset_id:
            message["dataset_id"] = dataset_id
        if video_file:
            message["video_file"] = video_file
        if error:
            message["error"] = str(error)

        channel.basic_publish(exchange='', routing_key=RABBITMQ_QUEUE_NAME, body=json.dumps(message))
        connection.close()
        print(f"✅ Sent RabbitMQ Message: {message}")
    except Exception as e:
        print(f"❌ Failed to send RabbitMQ message: {e}")

class XAIRequest(BaseModel):
    dataset_id: str
    algorithms: List[str]

async def async_http_post(url, json_data=None, files=None):
    """Make asynchronous POST requests to a given URL with JSON data or files."""
    async with aiohttp.ClientSession() as session:
        if json_data:
            response = await session.post(url, json=json_data)
        elif files:
            response = await session.post(url, data=files)
        else:
            response = await session.post(url)

        if response.status != 200:
            logging.error(f"❌ Error in POST request to {url}: {response.status} - {await response.text()}")
            raise HTTPException(status_code=response.status, detail=await response.text())

        return await response.json()

async def download_dataset(dataset_id: str) -> str:
    """Download the dataset and return the local dataset path."""
    try:
        local_dataset_path = f"dataprocess/datasets/{dataset_id}"
        return local_dataset_path
    except Exception as e:
        logging.error(f"❌ Error downloading dataset {dataset_id}: {e}")
        raise

async def run_xai_process(dataset_id: str, algorithm_names: List[str]):
    """Run the XAI process asynchronously."""
    try:
        local_dataset_path = await download_dataset(dataset_id)
        dataset_dirs = [local_dataset_path]

        # Convert algorithm names to function references
        selected_algorithms = [cam_resnet.CAM_ALGORITHMS_MAPPING[name] for name in algorithm_names]

        # Run the XAI method
        cam_resnet.xai_run(dataset_dirs, selected_algorithms)

        # 🔄 Log XAI Processing in Neo4j
        provenance.create_processing_step("XAI Analysis", "XAI", f"XAI applied on {dataset_id} using {algorithm_names}")
        provenance.link_dataset_to_processing(dataset_id, "XAI Analysis")

        # 🚀 Notify RabbitMQ
        send_rabbitmq_message("xai_processing_completed", dataset_id)

    except Exception as e:
        logging.error(f"❌ Error in run_xai_process: {e}")
        send_rabbitmq_message("xai_processing_failed", dataset_id, error=str(e))
        raise

@app.post("/cam_xai")
async def run_xai(request: XAIRequest, background_tasks: BackgroundTasks):
    """Trigger XAI processing asynchronously."""
    try:
        background_tasks.add_task(run_xai_process, request.dataset_id, request.algorithms)

        # 🚀 Notify RabbitMQ
        send_rabbitmq_message("xai_processing_started", request.dataset_id)

        return {"message": "✅ XAI processing for dataset has started successfully."}
    except Exception as e:
        logging.error(f"❌ Error in run_xai endpoint: {e}") 
        send_rabbitmq_message("xai_processing_failed", request.dataset_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

extractor = AttentionExtractor("facebook/timesformer-base-finetuned-k400")

class VideoExplainRequest(BaseModel):
    video_path: str
    num_frames: int = 8

@app.post("/staa-video-explain/")
async def staa_video_explain(request: VideoExplainRequest):
    """Perform video explanation using STAA."""
    try:
        video_path = request.video_path
        num_frames = request.num_frames

        # Extract STAA-based attention
        spatial_attention, temporal_attention, frames, logits = staa_model.extract_attention(video_path, num_frames)
        prediction_idx = torch.argmax(logits, dim=1).item()
        prediction = staa_model.model.config.id2label[prediction_idx]

        # 🔄 Log Explanation in Neo4j
        provenance.create_model_prediction(video_path, prediction)
        provenance.link_processing_to_prediction("XAI Analysis", video_path)

        # 🚀 Notify RabbitMQ
        send_rabbitmq_message("staa_explanation_completed", video_file=video_path)

        return {
            "prediction": prediction,
            "spatial_attention": spatial_attention.tolist(),
            "temporal_attention": temporal_attention.tolist(),
        }
    except Exception as e:
        logging.error(f"❌ Error in staa_video_explain: {e}")
        send_rabbitmq_message("staa_explanation_failed", video_file=request.video_path, error=str(e))
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
