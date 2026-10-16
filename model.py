from huggingface_hub import hf_hub_download
model_path=hf_hub_download(repo_id="iam-tsr/yolov8n-helmet-detection", filename = "best.pt",
local_dir="models",
)
print("model dowloaded successfully!")
print("Model path:", model_path)