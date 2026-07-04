:

🎙️ Hidden Emotion Detection Using Voice
## Project Summary

Developed an AI-powered speech emotion recognition system capable of identifying both explicit emotions expressed by a speaker and hidden emotional states that may be suppressed during speech. The solution leverages transformer-based speech representations, multi-stage training, and a dual-path attention architecture to analyze emotional characteristics from audio signals. The framework is trained using the IEMOCAP and CREMA-D datasets and validated through statistical analysis based on inter-annotator disagreement, enabling hidden emotion detection without requiring dedicated hidden emotion labels.

## Problem Statement

Human speech often conveys two distinct emotional layers:

Explicit Emotion: The emotion directly expressed by the speaker.
Hidden Emotion: The underlying emotional state that may be intentionally or unintentionally suppressed.

Traditional speech emotion recognition models focus only on explicit emotions. This project extends conventional SER by introducing a mechanism capable of estimating suppressed emotional states through differences in learned attention representations.

## Objectives
Develop a speech emotion recognition framework using transformer-based speech embeddings.
Detect both explicit and hidden emotions from speech.
Improve contextual understanding using dual-path attention.
Validate hidden emotion estimation without ground-truth hidden emotion labels.
Build a scalable framework for future real-time emotion-aware applications.
System Architecture

(Keep your existing architecture diagram exactly as it is.)

## Methodology
**Stage 1 — Supervised Fine-Tuning**

The pretrained HuBERT Base (facebook/hubert-base-ls960) model is fine-tuned on the IEMOCAP and CREMA-D datasets to classify four primary emotions: Angry, Happy, Neutral, and Sad. Class imbalance is handled using WeightedRandomSampler, while CosineAnnealingLR and label smoothing improve training stability.

**Stage 2 — Self-Supervised Adapter Learning**

Instead of retraining the complete encoder, lightweight bottleneck adapter layers are introduced into every transformer block. The HuBERT encoder remains frozen while only adapter parameters are optimized using masked waveform reconstruction, enabling efficient domain adaptation.

**Stage 3 — Hidden Emotion Prediction**

The final stage introduces two independent attention branches:

Global Attention Head
Local Window-Based Attention Head

The global branch focuses on the overall emotional representation of the utterance, while the local branch captures subtle temporal variations that may correspond to suppressed emotions.

The disagreement between these branches is quantified using KL Divergence, which serves as the suppression score.

**Key Features**
Three-stage deep learning architecture
Transformer-based speech representation using HuBERT
Parameter-Efficient Fine-Tuning (PEFT)
Dual-path attention for explicit and hidden emotion modeling
Statistical validation using inter-annotator disagreement
End-to-end PyTorch implementation
GPU-accelerated training on Google Colab

**Technology Stack**
Category	Technologies
Programming Language	Python
Framework	PyTorch
Transformer Models	HuBERT
Feature Extraction	Wav2Vec2 Feature Extractor
Audio Processing	torchaudio
Machine Learning	scikit-learn
Statistical Analysis	SciPy
Visualization	Matplotlib
Dataset	IEMOCAP, CREMA-D
Development Environment	Google Colab GPU





## 🏗️ Architecture — 3-Stage Pipeline

```
Raw Audio (16kHz WAV)
        │
        ▼
┌───────────────────────────────────────────────┐
│  STAGE 1: HuBERT Supervised Fine-tuning        │
│  • facebook/hubert-base-ls960                  │
│  • IEMOCAP + CREMA-D (4-class: ang/hap/neu/sad)│
│  • WeightedRandomSampler for class balance     │
│  • Label smoothing (0.05), CosineAnnealingLR   │
│  → Saves: stage1_best.pt                       │
└────────────────────────┬──────────────────────┘
                         │ Frozen encoder weights
                         ▼
┌───────────────────────────────────────────────┐
│  STAGE 2: Self-Supervised Adapter Learning     │
│  • Bottleneck adapters (768→64→768) per layer  │
│  • Masked waveform reconstruction on CREMA-D  │
│  • Only adapter params trained (encoder frozen)│
│  • MSE loss on masked hidden frame regions     │
│  → Saves: stage2_adapters.pt                  │
└────────────────────────┬──────────────────────┘
                         │ Frozen encoder + adapters
                         ▼
┌───────────────────────────────────────────────┐
│  STAGE 3: Dual-Path Attention Training         │
│                                                │
│  Global Attention Path (utterance-level)       │
│  → Explicit Emotion Head                       │
│                                                │
│  Local Attention Path (window=4 frames)        │
│  → Hidden Emotion Head                         │
│                                                │
│  KL divergence between heads = Suppression     │
│  Loss = CE_exp + CE_hid + λ_orth·cosine_sim    │
│         − λ_div·KL                            │
│  → Saves: stage3_best.pt                       │
└───────────────────────────────────────────────┘
        │                        │
        ▼                        ▼
 Explicit Emotion          KL Divergence Score
 (what is expressed)       (suppression signal)
```

---

## ✨ Key Innovations

**Dual-Path Attention** — Two parallel attention mechanisms run on the same encoder output. The global path (single learnable query across full utterance) captures expressed emotion; the local path (windowed QKV with frame window=4) captures frame-level micro-expressions and suppressed states.

**KL Divergence as Suppression Score** — The divergence between the two attention heads' softmax distributions is the suppression signal. High KL = the heads disagree = likely emotional suppression.

**Label-Free Validation** — No hidden emotion ground truth labels exist in any public dataset. The system is validated using IEMOCAP inter-annotator disagreement entropy as a proxy: where human raters disagreed on an utterance, the model should flag suppression.

**Composite Training Loss** — 
```
Loss = CE(explicit) + 1.0·CE(hidden) + 0.1·cosine_similarity − 0.1·KL
```
The orthogonality regularizer pushes the two attention paths to learn different representations. The negative KL term maximizes their divergence — forcing the hidden head to find an alternate emotional interpretation.


## 📊 Results

### Explicit Emotion Classification (IEMOCAP Session 5)

| Emotion | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Angry | 0.69 | 0.71 | 0.70 | 170 |
| Happy | 0.70 | 0.60 | 0.65 | 442 |
| Neutral | 0.62 | 0.73 | 0.67 | 384 |
| Sad | 0.73 | 0.69 | 0.71 | 245 |
| **Weighted Avg** | **0.68** | **0.67** | **0.67** | **1241** |

**Overall Accuracy: 67%** on 4-class emotion classification

### Hidden Emotion Detection Validation

| Metric | Value |
|---|---|
| AUC-ROC | 0.5754 |
| Mann-Whitney p-value | 0.0001 (statistically significant) |
| Head disagreement (ambiguous utterances) | 0.072 |
| Head disagreement (clear utterances) | 0.047 |
| Best F1 (75th percentile KL threshold) | 0.292 |
| Utterances flagged as suppressed | 4.4% (55/1241) |

### Top Suppression Patterns (Expressed → Hidden)
- happy → neutral: **31.6%** (most common suppression)
- sad → neutral: **21.1%**
- angry → happy / neutral / angry → neutral: ~10.5% each

---
## Application
Mental health monitoring
Human–Computer Interaction
Intelligent Virtual Assistants
Customer sentiment analysis
Call center analytics
Healthcare decision support
Emotion-aware conversational AI
Educational technology

## Future Scope
Real-time speech emotion detection
Multilingual emotion recognition
Cross-dataset generalization
Deployment using FastAPI
Interactive Gradio web application
Integration with Large Language Models (LLMs)
## 🚀 Getting Started

### 1. Clone & Open in Colab

```bash
git clone https://github.com/<your-username>/hidden-emotion-detection.git
```

Open `Hidden_Emotion_Detection_voice.ipynb` in [Google Colab](https://colab.research.google.com/) with GPU runtime.

### 2. Dataset Setup

Both datasets are downloaded via Kaggle API. Upload your `kaggle.json` when prompted:

```python
# IEMOCAP (~12hrs dyadic speech, 10 actors)
kagglehub.dataset_download("dejolilandry/iemocapfullrelease")

# CREMA-D (7442 clips, 91 actors, ages 20-74)
kagglehub.dataset_download("ejlok1/cremad")
```

### 3. Install Dependencies

```bash
pip install transformers datasets torchaudio scikit-learn pandas kaggle
```

### 4. Run Cells in Order

| Cell | What it does |
|---|---|
| Cell 1–2 | Install + download datasets |
| Cell 3 | Config (device, sample rate, emotion classes) |
| Cell 4 | Model definitions (AdapterLayer, DualPathAttention, HiddenEmotionModel) |
| Cell 5–7 | Dataset loaders + balanced sampler |
| Cell 8 | Stage 1: HuBERT supervised fine-tuning |
| Cell 9 | Stage 2: SSL adapter learning |
| Cell 10 | Stage 3: Dual-path attention training |
| Cell 11 | Final evaluation + hidden emotion analysis |
| Cell 12 | Annotator disagreement validation (3 checks) |

---

## ⚙️ Configuration

```python
SAMPLE_RATE  = 16000
MAX_LENGTH   = 6 * SAMPLE_RATE   # 6-second clips
BATCH_SIZE   = 16
EPOCHS_S1    = 8                  # Stage 1 supervised fine-tuning
EPOCHS_SSL   = 3                  # Stage 2 adapter SSL
EPOCHS_S3    = 8                  # Stage 3 dual-path attention
GRAD_ACCUM   = 4                  # Effective batch = 64
EMOTIONS     = ['angry', 'happy', 'neutral', 'sad']  # frustrated excluded
```

**Why `frustrated` was dropped:** Poor inter-annotator agreement, not present in CREMA-D, consistently lowest F1. Dropping it produces a cleaner 4-class model aligned with most literature.

---

## 🔍 Validation Methodology

Since no ground truth labels for hidden emotions exist, validation uses a 3-check framework:

**Check A — Detection Validity:** Does the KL score fire on utterances that human raters found ambiguous? Measured via AUC-ROC and Mann-Whitney U test against inter-annotator entropy.

**Check B — Head Disagreement:** On high-entropy utterances, does the explicit head predict the majority label while the hidden head predicts a different emotion? The "smoking gun" cases (19 utterances) confirm this pattern.

**Check C — Distribution Shift:** Does the hidden head's emotion distribution on ambiguous utterances differ from clear ones? Neutral predictions dominate ambiguous utterances (49%), suggesting neutral is the default mask for suppressed emotions.

---

## Hidden Emotion Detection Using Voice

## My Contributions

Contributed to the end-to-end development of the speech emotion recognition pipeline using HuBERT, Wav2Vec2, and PyTorch.
Participated in audio preprocessing, model training, feature extraction, and evaluation using the IEMOCAP and CREMA-D datasets.
Assisted in implementing and validating the dual-path attention mechanism for improved hidden emotion detection.
Collaborated with the team in debugging, performance optimization, experimentation, and documentation.

## 📂 Project Structure

```
.
├── Hidden_Emotion_Detection_voice.ipynb   # Main notebook
├── stage1_best.pt                         # Stage 1 checkpoint (auto-saved)
├── stage2_adapters.pt                     # Stage 2 adapter weights
├── stage3_best.pt                         # Final model checkpoint
├── hidden_emotion_validation.png          # 3-panel validation figure
├── kl_validation_results.csv             # Per-utterance results
└── README.md
```

---

## 🗺️ Roadmap

- [ ] Integrate MSP-Podcast for richer SSL (60hrs natural speech, free academic license)
- [ ] Real-time inference on microphone input
- [ ] Gradio demo for audio upload + emotion visualization
- [ ] Extend to more emotions (fear, disgust) using additional datasets
- [ ] Cross-corpus evaluation on MELD or MSP-IMPROV

---

## 🙏 Acknowledgements

- IEMOCAP dataset: USC Institute for Creative Technologies
- CREMA-D: Houwei Cao et al., IEEE TASLP 2014
- HuBERT: [facebook/hubert-base-ls960](https://huggingface.co/facebook/hubert-base-ls960)
- Mentored by Prof. Subir Roy, RV University Bangalore

---

## 📄 License

MIT License
