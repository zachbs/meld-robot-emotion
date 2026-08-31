import os
import time
from urllib import response
import cv2
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoTokenizer, AutoModel, CLIPProcessor, CLIPModel, AutoModelForCausalLM
from llama_cpp import Llama
import torch
torch.set_num_threads(4) # Match physical CPU cores

# ---------------------------------------------------------
# Configuration & Model Paths
# ---------------------------------------------------------
CLASSIFIER_PATH = "emotion_classifier_best.pth"

# If downloaded via Hugging Face hub:
MODEL_ID = r"./models/Qwen2.5-1.5B-Instruct"

# If stored in a specific local folder path, use a raw string:
# MODEL_ID = r"C:\Users\zbsmi\.cache\huggingface\hub\models--Qwen--Qwen2.5-1.5B-Instruct\snapshots\<hash>"

EMOTION_MAP = {
    0: "Neutral",
    1: "Joy",
    2: "Surprise",
    3: "Anger",
    4: "Sadness",
    5: "Disgust",
    6: "Fear"
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------
# Multimodal Classifier Architecture
# ---------------------------------------------------------
class MultimodalEmotionClassifier(nn.Module):
    def __init__(self, input_dim=1280, num_classes=7):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.50),
            
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.35),
            
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.network(x)

# ---------------------------------------------------------
# Pipeline Initializer
# ---------------------------------------------------------
class EmotionResponsePipeline:
    def __init__(self, classifier_path=CLASSIFIER_PATH, model_id=MODEL_ID):
        self.load_latencies = {}
        t_init_start = time.perf_counter()

        print(f"Initializing pipeline on device: {DEVICE}")
        
        # 1. Load RoBERTa
        t0 = time.perf_counter()
        print("Loading RoBERTa text encoder...")
        self.text_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-emotion")
        self.text_model = AutoModel.from_pretrained("cardiffnlp/twitter-roberta-base-emotion").to(DEVICE)
        self.text_model.eval()
        self._sync_device()
        self.load_latencies['roberta_load_ms'] = (time.perf_counter() - t0) * 1000.0

        # 2. Load CLIP
        t0 = time.perf_counter()
        print("Loading CLIP vision encoder...")
        self.vision_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.vision_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
        self.vision_model.eval()
        self._sync_device()
        self.load_latencies['clip_load_ms'] = (time.perf_counter() - t0) * 1000.0

        # 3. Load Custom Emotion Classifier
        t0 = time.perf_counter()
        print(f"Loading trained emotion classifier from '{classifier_path}'...")
        self.classifier = MultimodalEmotionClassifier(input_dim=1280, num_classes=7).to(DEVICE)
        self.classifier.load_state_dict(torch.load(classifier_path, map_location=DEVICE))
        self.classifier.eval()
        self._sync_device()
        self.load_latencies['classifier_load_ms'] = (time.perf_counter() - t0) * 1000.0

        # 4. Load Qwen LLM
        t0 = time.perf_counter()
        print(f"Loading Qwen model ({model_id})...")
        self.llm_tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.llm = Llama(
            model_path="./models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
            n_ctx=512,
            n_threads=4, # Matches Ryzen 3500U physical cores
            verbose=False
        )
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
        self._sync_device()
        self.load_latencies['llm_load_ms'] = (time.perf_counter() - t0) * 1000.0

        self.load_latencies['total_model_loading_ms'] = (time.perf_counter() - t_init_start) * 1000.0

    def _sync_device(self):
        """Ensures accurate timing measurements when executing on CUDA GPUs."""
        if DEVICE == "cuda":
            torch.cuda.synchronize()

    def extract_middle_frame(self, video_path):
        """Extracts the center frame from an MP4 video clip."""
        if not os.path.exists(video_path):
            return None
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return None
        
        target_frame = total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        cap.release()
        
        if ret and frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame_rgb)
        return None

    def compute_text_embedding(self, text):
        inputs = self.text_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEVICE)
        with torch.no_grad():
            outputs = self.text_model(**inputs)
            attention_mask = inputs['attention_mask'].unsqueeze(-1)
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            text_emb = sum_embeddings / sum_mask
        return text_emb

    def compute_vision_embedding(self, image):
        if image is None:
            return torch.zeros((1, 512), device=DEVICE)
        
        inputs = self.vision_processor(images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = self.vision_model.get_image_features(**inputs)
            if isinstance(outputs, torch.Tensor):
                vision_emb = outputs
            elif hasattr(outputs, "image_embeds"):
                vision_emb = outputs.image_embeds
            elif hasattr(outputs, "pooler_output"):
                vision_emb = outputs.pooler_output
            else:
                vision_emb = outputs[0]

            vision_emb = vision_emb / vision_emb.norm(p=2, dim=-1, keepdim=True)
        return vision_emb

    def generate_llm_response(self, text, emotion):
        messages = [
            {
                "role": "system",
                "content": "You are an empathetic, emotionally intelligent assistant. Respond appropriately to the user based on their statement and perceived emotional state."
            },
            {
                "role": "user",
                "content": f"User Said: \"{text}\"\nDetected Emotion: {emotion}\n\nProvide an empathetic response:"
            }
        ]

        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=50,
            temperature=0.7,
            top_p=0.9
        )

        return response["choices"][0]["message"]["content"].strip()

    def run(self, text, video_path=None):
        latencies = {}
        t_total_start = time.perf_counter()

        # 1. Video Frame Extraction Latency
        t0 = time.perf_counter()
        frame = self.extract_middle_frame(video_path) if video_path else None
        self._sync_device()
        latencies['frame_extraction_ms'] = (time.perf_counter() - t0) * 1000.0

        # 2. RoBERTa Text Encoding Latency
        t0 = time.perf_counter()
        text_emb = self.compute_text_embedding(text)
        self._sync_device()
        latencies['text_encoding_ms'] = (time.perf_counter() - t0) * 1000.0

        # 3. CLIP Vision Encoding Latency
        t0 = time.perf_counter()
        vision_emb = self.compute_vision_embedding(frame)
        self._sync_device()
        latencies['vision_encoding_ms'] = (time.perf_counter() - t0) * 1000.0

        # 4. MLP Emotion Classifier Latency
        t0 = time.perf_counter()
        combined_emb = torch.cat([text_emb, vision_emb], dim=-1)  # Shape: (1, 1280)
        with torch.no_grad():
            logits = self.classifier(combined_emb)
            probs = torch.softmax(logits, dim=-1)
            pred_idx = logits.argmax(dim=-1).item()
            confidence = probs[0, pred_idx].item()

        predicted_emotion = EMOTION_MAP.get(pred_idx, "Unknown")
        self._sync_device()
        latencies['classifier_inference_ms'] = (time.perf_counter() - t0) * 1000.0

        # 5. Qwen Response Generation Latency
        t0 = time.perf_counter()
        llm_response = self.generate_llm_response(text, predicted_emotion)
        self._sync_device()
        latencies['llm_generation_ms'] = (time.perf_counter() - t0) * 1000.0

        latencies['total_inference_ms'] = (time.perf_counter() - t_total_start) * 1000.0

        return {
            "input_text": text,
            "predicted_emotion": predicted_emotion,
            "confidence": confidence,
            "llm_response": llm_response,
            "latencies": latencies
        }


# ---------------------------------------------------------
# Execution Test & Benchmark Bench
# ---------------------------------------------------------
if __name__ == "__main__":
    script_start_time = time.perf_counter()

    test_text = "Oh anyway, I just wanted to say how wonderful I think you are."
    test_video_path = "data/MELD.Raw/output_repeated_splits_test/dia240_utt7.mp4"

    # 1. Measure Model Loading Latency
    print("================ MODEL INITIALIZATION ================")
    pipeline = EmotionResponsePipeline()

    # 2. Warmup Pass (Executes CUDA memory allocation and graph build)
    print("\nExecuting warm-up pass...")
    _ = pipeline.run(text=test_text, video_path=test_video_path)

    # 3. Measure Steady-State Inference Latency
    print("Executing benchmark inference pass...")
    result = pipeline.run(text=test_text, video_path=test_video_path)

    total_script_time_ms = (time.perf_counter() - script_start_time) * 1000.0

    print("\n================ PIPELINE RESULT ================")
    print(f"Input Text:        {result['input_text']}")
    print(f"Predicted Emotion: {result['predicted_emotion']} (Confidence: {result['confidence']*100:.2f}%)")
    print(f"LLM Response:      {result['llm_response']}")
    
    print("\n================ 1. MODEL LOADING LATENCY ================")
    for stage, duration in pipeline.load_latencies.items():
        print(f"{stage:<25}: {duration/1000.0:8.2f} s ({duration:10.2f} ms)")

    print("\n================ 2. STEADY-STATE INFERENCE LATENCY ================")
    for stage, duration in result['latencies'].items():
        print(f"{stage:<25}: {duration:8.2f} ms")

    print("\n================ 3. TOTAL OVERALL EXECUTION TIME ================")
    print(f"Total Wall-Clock Time    : {total_script_time_ms/1000.0:8.2f} s ({total_script_time_ms:10.2f} ms)")