# Real-Time Multimodal Emotion-Aware Response Pipeline

A real-time, end-to-end multimodal pipeline designed for **emotion-aware robotic character interactions** using the **Multimodal EmotionLines Dataset (MELD)**.

The system processes conversational text and video frames, extracts multimodal feature embeddings, classifies the underlying emotion into structured tags, and generates contextually empathetic text responses.

---

## Overview & Track Selection

- **Selected Track:** Text + Vision
- **Primary Objective:** Transform conversational text and synchronized video frames into real-time structured emotion states and empathetic response generation.
- **Dataset Foundation:** [Multimodal EmotionLines Dataset (MELD)](https://affective-meld.github.io/)

MELD provides conversational dialogue data across **7 emotion categories**:

- Neutral
- Joy
- Surprise
- Anger
- Sadness
- Disgust
- Fear

---

## System Architecture

The pipeline uses a modular, multi-stage processing architecture:

```text
┌─────────────────────┐
│     Input Text      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────┐
│ RoBERTa Text Encoder        │
│ 768-Dimensional Embedding   │
└────────────┬────────────────┘
             │
             │
             │       ┌─────────────────────┐
             │       │   Video Frame       │
             │       └──────────┬──────────┘
             │                  │
             │                  ▼
             │       ┌─────────────────────────────┐
             │       │ CLIP Vision Encoder         │
             │       │ 512-Dimensional Embedding   │
             │       └────────────┬────────────────┘
             │                    │
             └─────────┬──────────┘
                       ▼
          ┌──────────────────────────┐
          │ Feature Concatenation    │
          │ 1280-Dimensional Vector  │
          └────────────┬─────────────┘
                       │
                       ▼
          ┌──────────────────────────┐
          │ Multimodal MLP           │
          │ Emotion Classifier       │
          └────────────┬─────────────┘
                       │
                       ▼
          ┌──────────────────────────┐
          │ Emotion Tag + Confidence  │
          └────────────┬─────────────┘
                       │
                       ▼
          ┌──────────────────────────┐
          │ Qwen2.5-1.5B-Instruct    │
          │ LLM Response Generation  │
          └────────────┬─────────────┘
                       │
                       ▼
          ┌──────────────────────────┐
          │ Empathetic Response      │
          └──────────────────────────┘
```

### Pipeline Stages

1. **Vision Preprocessing**
   - Extracts the middle frame of an MP4 video clip.
   - Uses the middle frame to capture the core facial expression during an utterance.

2. **Text Feature Extraction**
   - Encodes conversational text using `cardiffnlp/twitter-roberta-base-emotion`.
   - Produces a **768-dimensional mean-pooled representation**.

3. **Vision Feature Extraction**
   - Processes the extracted frame using `openai/clip-vit-base-patch32`.
   - Produces a normalized **512-dimensional visual embedding**.

4. **Multimodal Fusion & Classification**
   - Concatenates the text and vision embeddings into a **1280-dimensional feature vector**.
   - Passes the combined representation through a custom 3-layer PyTorch MLP classifier.
   - Outputs emotion logits and a confidence score.

5. **Grounded Response Generation**
   - Passes the original utterance and predicted emotion tag to `Qwen/Qwen2.5-1.5B-Instruct`.
   - The LLM uses a system prompt designed to generate concise, contextually appropriate, and empathetic responses.

---

## Models

| Component | Model | Output / Role |
|---|---|---|
| **Text Encoder** | `cardiffnlp/twitter-roberta-base-emotion` | 768-D text embedding |
| **Vision Encoder** | `openai/clip-vit-base-patch32` | 512-D image embedding |
| **Classifier** | `MultimodalEmotionClassifier` | Emotion classification |
| **LLM** | `Qwen/Qwen2.5-1.5B-Instruct` | Empathetic response generation |

---

## Parameter Budget & Constraint Verification

The complete local inference pipeline is constrained to **6 billion or fewer learned parameters** across all models.

| Component | Model Backbone | Approx. Parameters |
|---|---|---:|
| **Text Encoder** | `cardiffnlp/twitter-roberta-base-emotion` | ~125M |
| **Vision Encoder** | `openai/clip-vit-base-patch32` | ~151M |
| **Classifier Head** | `MultimodalEmotionClassifier` | ~0.72M |
| **LLM Generator** | `Qwen/Qwen2.5-1.5B-Instruct` | ~1.54B |
| **Total System** | Combined Pipeline | **~1.82B** |

> **Status: PASS**
>
> The complete pipeline contains approximately **1.82 billion learned parameters**, which is well within the **6.0 billion parameter limit**.

---

## Hardware & Performance Benchmarks

### Environment

| Specification | Value |
|---|---|
| **Operating System** | Windows 11 |
| **GPU** | `[ADD YOUR GPU HERE]` |
| **VRAM Utilization** | `[ADD YOUR PEAK VRAM HERE]` |
| **Python** | Python 3.10+ |

### Real-Time Latency

For conversational robotics, **real-time interaction** is defined here as maintaining an end-to-end response delay below approximately **500 ms**.

| Pipeline Stage | Observed Latency |
|---|---:|
| **Frame Extraction** | `[ADD MS]` ms |
| **Text Encoding** | `[ADD MS]` ms |
| **Vision Encoding** | `[ADD MS]` ms |
| **Emotion Classification** | `[ADD MS]` ms |
| **Response Generation** | `[ADD MS]` ms |
| **Total End-to-End Latency** | `[ADD MS]` ms |

Latency measurements can be collected automatically using the timing instrumentation implemented in `pipeline.py`.

---

## Installation

### 1. Clone the Repository

```powershell
git clone https://github.com/your-username/meld-robot-emotion.git
cd meld-robot-emotion
```

### 2. Create a Virtual Environment

```powershell
python -m venv .venv
```

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

Install PyTorch:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Install the remaining dependencies:

```powershell
pip install transformers clip-by-openai opencv-python pillow llama-cpp-python
```

> **Note:** If `llama-cpp-python` cannot be compiled successfully on your system, the pipeline can alternatively use the standard Hugging Face `transformers` implementation of Qwen2.5-1.5B.

---

## Pretrained Weights

Place the trained multimodal classification head in the project root:

```text
emotion_classifier_best.pth
```

If using the GGUF implementation of the LLM, place the model at:

```text
.cache/
└── huggingface/
    └── download/
        └── qwen2.5-1.5b-instruct-q4_k_m.gguf
```

Alternatively, update the `GGUF_PATH` variable in `pipeline.py` to point to the location of the GGUF model.

---

## Project Structure

```text
meld-robot-emotion/
│
├── pipeline.py
├── emotion_classifier_best.pth
├── README.md
│
├── data/
│   └── MELD.Raw/
│       ├── dev_splits_complete/
│       ├── train_splits/
│       └── test_splits_complete/
│
├── .cache/
│   └── huggingface/
│       └── download/
│           └── qwen2.5-1.5b-instruct-q4_k_m.gguf
│
└── .venv/
```

---

## Running the Pipeline

Run the complete end-to-end pipeline with:

```powershell
python pipeline.py
```

The pipeline will:

1. Load the input text.
2. Extract the middle frame from the input video.
3. Generate the RoBERTa text embedding.
4. Generate the CLIP vision embedding.
5. Concatenate the multimodal features.
6. Classify the user's emotion.
7. Generate an empathetic response using Qwen2.5-1.5B.
8. Report the latency of each pipeline stage.

---

## Example Input

### Text

```text
I can't believe you actually remembered my birthday!
```

### Video

```text
data/MELD.Raw/dev_splits_complete/dia0_utt0.mp4
```

### Example Output

```text
================ PIPELINE RESULT ================

Input Text:        I can't believe you actually remembered my birthday!
Predicted Emotion: Joy (Confidence: 94.20%)

LLM Response:
That makes me so happy to hear! Birthdays are special, and I wanted to make sure you felt remembered today.

================ LATENCY BREAKDOWN ================

frame_extraction_ms      :    12.40 ms
text_encoding_ms         :    15.20 ms
vision_encoding_ms       :    18.10 ms
classifier_inference_ms  :     1.50 ms
llm_generation_ms        :   210.30 ms
total_pipeline_ms        :   257.50 ms
```

---

## Architectural Decisions

### Middle-Frame Extraction

Rather than processing the entire video with a 3D CNN or video transformer, the pipeline extracts a single middle frame.

**Advantages:**

- Significantly reduces visual computation.
- Requires only one CLIP inference pass.
- Reduces memory usage.
- Helps maintain low end-to-end latency.

**Trade-off:**

The system may miss micro-expressions or important facial changes occurring exclusively at the beginning or end of an utterance.

---

### Early Multimodal Fusion

The system uses feature-level concatenation:

```text
RoBERTa: 768-D
        +
CLIP:    512-D
        =
Combined: 1280-D
```

The resulting 1280-dimensional representation is passed into the multimodal MLP classifier.

This approach provides a lightweight method for allowing the classifier to learn relationships between linguistic and visual features without introducing a significantly larger multimodal architecture.

---

### Quantized LLM Generation

The pipeline supports a **4-bit GGUF version** of Qwen2.5-1.5B for local inference.

Using a quantized model reduces memory requirements compared with FP16 inference and makes it more practical to run the LLM concurrently with the text and vision encoders on consumer hardware.

---

## Limitations

### Single-Frame Vision

Only one frame is currently extracted from each video clip.

This means the model does not explicitly model temporal facial dynamics or changes in expression throughout the utterance.

### No Audio Modality

The current implementation intentionally focuses on:

```text
Text + Vision
```

Audio-based emotion information is not currently included.

### Limited Conversational Context

The response generation stage primarily uses the current utterance and predicted emotion rather than maintaining a long conversational history.

### LLM Response Variability

Because the response is generated by an LLM, outputs may vary between runs and may occasionally produce responses that do not perfectly match the predicted emotion.

---

## Scope Transparency

### Completed

- [x] End-to-end multimodal Text + Vision pipeline
- [x] RoBERTa text feature extraction
- [x] CLIP visual feature extraction
- [x] Custom PyTorch multimodal classification head
- [x] MELD-based emotion classification
- [x] Seven-class emotion prediction
- [x] Qwen2.5-1.5B response generation
- [x] Emotion-grounded empathetic responses
- [x] Per-stage latency measurement
- [x] Total parameter budget calculation
- [x] Hugging Face LLM inference support
- [x] GGUF / `llama-cpp-python` inference support

### Intentionally Excluded

- [ ] **Audio Modality Integration**
  - Whisper / wav2vec-based audio processing was excluded to prioritize low-latency Text + Vision inference.

- [ ] **RLHF / RLAIF**
  - Reinforcement learning-based LLM fine-tuning was excluded to focus development on multimodal emotion classification and response latency.

- [ ] **Live Camera / Microphone Streaming**
  - The current implementation processes video clips rather than providing a live WebRTC, ROS 2, or camera/microphone streaming interface.

---

## Evaluation Metrics

The following metrics should be reported before final submission:

| Metric | Result |
|---|---:|
| **Emotion Classification Accuracy** | `[ADD VALUE]` |
| **Macro F1 Score** | `[ADD VALUE]` |
| **Frame Extraction Latency** | `[ADD MS]` |
| **Text Encoding Latency** | `[ADD MS]` |
| **Vision Encoding Latency** | `[ADD MS]` |
| **Classifier Latency** | `[ADD MS]` |
| **LLM Generation Latency** | `[ADD MS]` |
| **Total Pipeline Latency** | `[ADD MS]` |
| **Peak VRAM Usage** | `[ADD VALUE]` |
| **Total Learned Parameters** | **~1.82B** |

---

## Requirements Before Submission

The following values should be updated with measurements from the target hardware:

1. **GPU**
   - Replace `[ADD YOUR GPU HERE]` with the GPU used for evaluation.

2. **Peak VRAM**
   - Replace `[ADD YOUR PEAK VRAM HERE]` with the maximum observed VRAM usage.

3. **Latency Measurements**
   - Replace each `[ADD MS]` placeholder with the values reported by `pipeline.py`.

4. **Classification Performance**
   - Add validation/test **accuracy** and **Macro F1** if available.

---

## Key Takeaways

The resulting architecture provides a lightweight multimodal emotion-aware response system by combining:

```text
Conversational Text
        │
        ▼
    RoBERTa
        │
        ├──────────────┐
        │              │
        │          Video Frame
        │              │
        │              ▼
        │             CLIP
        │              │
        └──────┬───────┘
               ▼
       Multimodal MLP
               │
               ▼
        Emotion + Confidence
               │
               ▼
        Qwen2.5-1.5B
               │
               ▼
      Empathetic Response
```

With approximately **1.82B total learned parameters**, the system remains well below the **6B parameter constraint** while providing a complete local multimodal inference pipeline suitable for emotion-aware conversational robotics.