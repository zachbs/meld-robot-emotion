import os
import cv2
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, CLIPProcessor, CLIPModel

# ---------------------------------------------------------
# Configuration & File Paths
# ---------------------------------------------------------

# this chooses which dataset to create train = 0, val = 1, test = 2
dataset = 2


if dataset == 0:
    CSV_PATH = "data/MELD.Raw/train_sent_emo.csv"
    VIDEO_DIR = "data/MELD.Raw/train_splits"
    OUTPUT_PATH = "data/cached_features_train.pt"
elif dataset == 1:
    CSV_PATH = "data/MELD.Raw/dev_sent_emo.csv"
    VIDEO_DIR = "data/MELD.Raw/dev_splits_complete"
    OUTPUT_PATH = "data/cached_features_dev.pt"
else:
    CSV_PATH = "data/MELD.Raw/test_sent_emo.csv"
    VIDEO_DIR = "data/MELD.Raw/output_repeated_splits_test"
    OUTPUT_PATH = "data/cached_features_test.pt"

EMOTION_MAP = {
    'neutral': 0,
    'joy': 1,
    'surprise': 2,
    'anger': 3,
    'sadness': 4,
    'disgust': 5,
    'fear': 6
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using compute device: {DEVICE}")

# ---------------------------------------------------------
# Load Pre-trained Encoders
# ---------------------------------------------------------
print("Loading RoBERTa text encoder...")
text_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-emotion")
text_model = AutoModel.from_pretrained("cardiffnlp/twitter-roberta-base-emotion").to(DEVICE)
text_model.eval()

print("Loading CLIP vision encoder...")
vision_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
vision_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
vision_model.eval()

# ---------------------------------------------------------
# Feature Extraction Helpers
# ---------------------------------------------------------
def extract_middle_frame(video_path):
    """Opens MP4, calculates target middle frame, and converts to PIL RGB Image."""
    if not os.path.exists(video_path):
        return None
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None
    
    # Jump directly to the center frame timestamp
    target_frame = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    cap.release()
    
    if ret and frame is not None:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)
    return None

def compute_text_embedding(text):
    """Outputs a 768-dim mean-pooled RoBERTa text tensor."""
    inputs = text_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEVICE)
    with torch.no_grad():
        outputs = text_model(**inputs)
        attention_mask = inputs['attention_mask'].unsqueeze(-1)
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        text_emb = sum_embeddings / sum_mask
    return text_emb.squeeze(0).cpu()  # Shape: (768,)

def compute_vision_embedding(image):
    """Outputs a 512-dim normalized CLIP vision tensor."""
    if image is None:
        # Zero-vector fallback if video clip is missing or unreadable
        return torch.zeros(512)
    
    inputs = vision_processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = vision_model.get_image_features(**inputs)
        
        # Extract raw tensor if Hugging Face returns a ModelOutput dataclass
        if isinstance(outputs, torch.Tensor):
            vision_emb = outputs
        elif hasattr(outputs, "image_embeds"):
            vision_emb = outputs.image_embeds
        elif hasattr(outputs, "pooler_output"):
            vision_emb = outputs.pooler_output
        else:
            vision_emb = outputs[0]

        vision_emb = vision_emb / vision_emb.norm(p=2, dim=-1, keepdim=True)
    return vision_emb.squeeze(0).cpu()  # Shape: (512,)

# ---------------------------------------------------------
# Processing Pipeline
# ---------------------------------------------------------
def process_and_cache_dataset():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df = pd.read_csv(CSV_PATH)
    cached_dataset = []

    print(f"Processing {len(df)} samples from {CSV_PATH}...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        dia_id = row['Dialogue_ID']
        utt_id = row['Utterance_ID']
        text_content = str(row['Utterance'])
        emotion_str = str(row['Emotion']).lower()

        # Locate associated video file (format: diaX_uttY.mp4)
        video_filename = f"dia{dia_id}_utt{utt_id}.mp4"
        video_path = os.path.join(VIDEO_DIR, video_filename)

        # 1. Extract 768-dim text embedding
        text_emb = compute_text_embedding(text_content)

        # 2. Extract 512-dim middle frame vision embedding
        frame = extract_middle_frame(video_path)
        vision_emb = compute_vision_embedding(frame)

        # 3. Concatenate into single 1280-dim feature vector [768 + 512]
        combined_emb = torch.cat([text_emb, vision_emb], dim=-1)

        # 4. Map emotion string to target class integer
        label = EMOTION_MAP.get(emotion_str, 0)

        # Append structured sample dictionary
        cached_dataset.append({
            "features": combined_emb,                           # Tensor shape: (1280,)
            "label": torch.tensor(label, dtype=torch.long),     # Target class integer (0-6)
            "dialogue_id": dia_id,
            "utterance_id": utt_id
        })

    # Save compiled dataset directly to data directory
    torch.save(cached_dataset, OUTPUT_PATH)
    print(f"\nSuccessfully saved {len(cached_dataset)} samples to '{OUTPUT_PATH}'.")
    print(f"Sample tensor shape: {cached_dataset[0]['features'].shape}")

if __name__ == "__main__":
    process_and_cache_dataset()