from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import Model_ResNet
import os
import torch
import datetime
import json
import pika  # RabbitMQ Integration
from attention_extractor import AttentionExtractor
from neo4j_client import ProvenanceModel  # Neo4j Integration

app = FastAPI(title="Model Service")

# Initialize STAA (Spatio-Temporal Attention)
attention_extractor = AttentionExtractor(
    model_name="facebook/timesformer-base-finetuned-k400",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Output directory
OUTPUT_DIR = "video_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 🔗 Initialize Neo4j Provenance Tracking
provenance = ProvenanceModel()

# 🚀 RabbitMQ Setup
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
RABBITMQ_QUEUE_NAME = os.getenv("RABBITMQ_QUEUE_NAME", "pipeline_tasks")

def send_rabbitmq_message(event, dataset_id=None, video_file=None, error=None):
    """Send RabbitMQ messages for model events."""
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

# 📌 Run ResNet Model
@app.post("/resnet/{dataset_id}/{perturbation_func_name}/{severity}")
async def run_resnet_background(dataset_id: str, perturbation_func_name: str, severity: int, background_tasks: BackgroundTasks):
    """Run ResNet asynchronously for a dataset."""
    dataset_path = f"dataprocess/datasets/{dataset_id}"
    perturbed_dataset_path = f"dataprocess/datasets/{dataset_id}_{perturbation_func_name}_{severity}"

    dataset_paths = [dataset_path, perturbed_dataset_path]
    
    # 🚀 Start Model Processing in Background
    background_tasks.add_task(Model_ResNet.model_run, dataset_paths)

    # 🔄 Log Processing in Neo4j
    provenance.create_processing_step(
        step_name="Model Processing",
        step_type="resnet",
        config=f"ResNet applied on {dataset_id} with perturbation {perturbation_func_name} (severity {severity})"
    )
    provenance.link_dataset_to_processing(dataset_id, "Model Processing")

    # 🚀 Notify RabbitMQ
    send_rabbitmq_message("resnet_started", dataset_id)

    return {
        "message": f"✅ ResNet model started for dataset {dataset_id} with perturbation {perturbation_func_name} (severity {severity})."
    }

# 📌 Video Explanation with Facebook Timesformer
@app.post("/facebook/timesformer-base-finetuned-k400/{dataset_id}")
async def video_explain(dataset_id: str):
    """Explain videos using the Facebook Timesformer model."""

    video_dir = "dataprocess/videos"
    video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]

    results = []
    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        try:
            # 🔄 Extract attention and logits
            spatial_attention, temporal_attention, frames, logits = attention_extractor.extract_attention(video_path)
            prediction_idx = torch.argmax(logits, dim=1).item()
            prediction = attention_extractor.model.config.id2label[prediction_idx]

            # Create a unique directory for each video’s result
            video_result_dir = os.path.join(OUTPUT_DIR, os.path.splitext(video_file)[0])
            os.makedirs(video_result_dir, exist_ok=True)

            # Save visualizations
            attention_extractor.visualize_attention(
                spatial_attention, temporal_attention, frames, video_result_dir, prediction, "Unknown"
            )

            # 🔄 Log Prediction in Neo4j
            provenance.create_model_prediction(video_file, prediction)
            provenance.link_processing_to_prediction("Model Processing", video_file)
            provenance.link_dataset_to_prediction(dataset_id, video_file)

            # 🚀 Notify RabbitMQ
            send_rabbitmq_message("video_explanation_completed", dataset_id, video_file)

            results.append({
                "video_file": video_file,
                "prediction": prediction,
                "results_dir": video_result_dir
            })
        except Exception as e:
            results.append({"video_file": video_file, "error": str(e)})
            send_rabbitmq_message("video_explanation_failed", dataset_id, video_file, error=str(e))

    return {"results": results}

# 📌 Run API Server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
