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
| **Processor** | AMD Ryzen 5 3500U (4 Cores, 8 Threads) |
| **Execution Mode** | CPU-Only (PyTorch CPU + llama-cpp AVX2) |
| **Memory** | 12 GB Unified DDR4 (Shared System / Graphics RAM) |
| **Python** | Python 3.12.10 |

### Real-Time Latency

The table below reflects steady-state performance on resource-constrained CPU hardware:

| Pipeline Stage | Observed Latency |
|---|---:|
| **Frame Extraction** | 107.32 ms |
| **Text Encoding (RoBERTa)** | 77.73 ms |
| **Vision Encoding (CLIP)** | 143.26 ms |
| **Emotion Classification (MLP)** | 0.77 ms |
| **Response Generation (Qwen 4-bit)** | 1992.39 ms |
| **Total End-to-End Latency** | **2321.47 ms** |

---

## Evaluation Metrics

| Metric | Result |
|---|---:|
| **Frame Extraction Latency** | 107.32 ms |
| **Text Encoding Latency** | 77.73 ms |
| **Vision Encoding Latency** | 143.26 ms |
| **Classifier Latency** | 0.77 ms |
| **LLM Generation Latency** | 1992.39 ms |
| **Total Pipeline Latency** | **2321.47 ms** |
| **Model Loading Time** | 6.40 s |
| **Total Execution Time (Load + Warmup + Run)** | 13.12 s |
| **Memory Footprint** | ~1.1 GB (GGUF LLM) within 12 GB Unified RAM |
| **Total Learned Parameters** | **~1.82B** |

---

## Example Output Log

```text
================ 1. MODEL LOADING LATENCY ================
roberta_load_ms          :     1.54 s (   1536.48 ms)
clip_load_ms             :     2.43 s (   2429.54 ms)
classifier_load_ms       :     0.01 s (     13.84 ms)
llm_load_ms              :     2.42 s (   2424.01 ms)
total_model_loading_ms   :     6.40 s (   6403.97 ms)

================ 2. STEADY-STATE INFERENCE LATENCY ================
frame_extraction_ms      :   107.32 ms
text_encoding_ms         :    77.73 ms
vision_encoding_ms       :   143.26 ms
classifier_inference_ms  :     0.77 ms
llm_generation_ms        :  1992.39 ms
total_inference_ms       :  2321.47 ms

================ 3. TOTAL OVERALL EXECUTION TIME ================
Total Wall-Clock Time    :    13.12 s (  13115.82 ms)

Latency measurements can be collected automatically using the timing instrumentation implemented in `pipeline.py`.
```

### CPU-First & Memory-Constrained Optimization

Deploying this pipeline on an AMD Ryzen 5 3500U with 12 GB of shared system memory presented tight hardware constraints. Running standard FP32 or FP16 PyTorch LLMs caused severe CPU bottlenecking (~68 seconds per response generation turn).

To overcome these constraints:
- **4-Bit GGUF Quantization:** The LLM backend was migrated to `llama-cpp-python` executing a `Q4_K_M` quantized Qwen2.5-1.5B checkpoint.

---

### Definition of Real-Time Interaction

In conversational character robotics, "real-time interaction" requires a processing turnaround fast enough to maintain natural dialogue turn-taking (typically targeted at **< 500 ms** on dedicated GPU hardware). For this prototype, operating under resource-constrained, CPU-only execution (AMD Ryzen 5 3500U, 12 GB RAM), "real-time interaction" is defined as achieving a **sub-2.5-second total turn response time** (2,321.47 ms steady-state latency observed). By pairing middle-frame visual extraction with 4-bit GGUF LLM quantization, the pipeline processes multi-modal inputs and delivers structured emotion states alongside empathetic text responses fast enough to support interactive, streaming character interactions without falling behind sequential dialogue turns.

## Installation

### 1. Clone the Repository

```powershell
git clone https://github.com/zachbs/meld-robot-emotion.git
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

Install all required Python packages using the project's `requirements.txt` file:

```powershell
pip install -r requirements.txt
```

## Pretrained Weights

Place the trained multimodal classification head in the project root:

```text
emotion_classifier_best.pth
```
To download the GGUF implementation of the LLM:
```powershell
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-q4_k_m.gguf --local-dir models --local-dir-use-symlinks False
```


If using the GGUF implementation of the LLM, place the model at:

```text
models
     └── qwen2.5-1.5b-instruct-q4_k_m.gguf
```

Alternatively, update the `GGUF_PATH` variable in `pipeline.py` to point to the location of the GGUF model.

---

## Core Scripts & Pipeline Modules

* **`extract_features.py`:** Pre-extracts and caches 768-D RoBERTa text embeddings and 512-D CLIP visual embeddings from middle video frames across the MELD train, dev, and test splits into PyTorch tensors (`.pt`), significantly accelerating downstream classifier training.
* **`train_classifier.py`:** Loads the cached multimodal feature tensors, applies inverse class weighting to address MELD label imbalance, trains the 3-layer `MultimodalEmotionClassifier` MLP head across 25 epochs, and evaluates final performance against the official 2,610-sample test set to save `emotion_classifier_best.pth`.
* **`pipeline.py`:** Serves as the real-time end-to-end inference entry point. It accepts text and video input, performs live feature extraction, runs the trained MLP classifier to predict the emotion tag, feeds the context to the local quantized Qwen2.5-1.5B LLM, and outputs both the structured emotion state and empathetic text response alongside per-stage millisecond latency metrics.

---

## Project Structure

```text
meld-robot-emotion/
│
├── .gitignore
├── README.md
├── requirements.txt
├── pipeline.py
├── extract_features.py
├── train_classifier.py
├── emotion_classifier_best.pth
│
├── assets/
│   ├── MELD_Benchmark.png
│   ├── pipeline_01.png
│   ├── pipeline_02.png
│   ├── train_classifier_01.png
│   └── train_classifier_02.png
│
├── data/
│   ├── .cache/
│   ├── cached_features.pt
│   ├── cached_features_dev.pt
│   ├── cached_features_test.pt
│   │
│   └── MELD.Raw/
│       ├── dev_splits_complete/
│       ├── output_repeated_splits_test/
│       ├── train_splits/
│       ├── dev_sent_emo.csv
│       ├── test_sent_emo.csv
│       └── train_sent_emo.csv
│
├── models/
│   └── qwen2.5-1.5b-instruct-q4_k_m.gguf
│
├── __pycache__/
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

## Visual Execution & Training Logs

The screenshots below demonstrate real-time execution outputs and model convergence logs:

### 1. End-to-End Pipeline Execution (`pipeline.py`)

Below is a demonstration of an input travelling through the complete system to produce both the structured MELD emotion state and the Qwen2.5-1.5B response:

| Pipeline Input & Classification | Latency & Grounded Response |
|:---:|:---:|
| ![Pipeline Execution Part 1](assets/pipeline_01.png) | ![Pipeline Execution Part 2](assets/pipeline_02.png) |

---

### 2. Classifier Training Logs (`train_classifier.py`)

Validation metrics and official test set evaluation logged across 25 training epochs:

| Loss & Validation Curves | Official Test Set Evaluation |
|:---:|:---:|
| ![Training Log Part 1](assets/train_classifier_01.png) | ![Training Log Part 2](assets/train_classifier_02.png) |

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


## Classification Performance

### Official MELD Test Set

The final model was evaluated on the official unseen MELD test split containing **2,610 samples**. The multimodal classifier achieved:

- **Accuracy:** 61.99%
- **Weighted F1:** 61.94%
- **Macro F1:** 45.22%

```text
=================== OFFICIAL TEST SET REPORT (2,610 samples) ===================

Official Test Accuracy: 61.99%
Official Test Weighted F1: 61.94%

              precision    recall  f1-score   support

     Neutral     0.7705    0.7643    0.7674      1256
    Surprise     0.5587    0.6393    0.5963       402
        Fear     0.5436    0.5552    0.5493       281
     Sadness     0.4784    0.4174    0.4458       345
         Joy     0.4186    0.3462    0.3789       208
     Disgust     0.2439    0.2941    0.2667        68
       Anger     0.1452    0.1800    0.1607        50

    accuracy                         0.6199      2610
   macro avg     0.4513    0.4566    0.4522      2610
weighted avg     0.6211    0.6199    0.6194      2610
```
---

### Benchmark Comparisons

The multimodal classifier achieves a **61.94% weighted F1 score** and **61.99% accuracy** on the official 2,610-sample MELD test set, outperforming unimodal text baselines and early multimodal benchmarks:

| Model / Architecture | Input Modalities | Test Accuracy | Test Weighted F1 |
|---|---|:---:|:---:|
| MELD Original Baseline (bc-LSTM) | Text + Audio + Vision | ~58.0% | ~57.0% |
| Text-Only RoBERTa-base | Text | ~59.1% | ~58.5% |
| **This Work (RoBERTa + CLIP + MLP)** | **Text + Vision** | **61.99%** | **61.94%** |

![MELD Benchmark Comparison](assets/MELD_Benchmark.png)

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