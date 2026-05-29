# 🛡️ RakshakAI
**Nepali Social Media Content Moderation System**

[![HuggingFace Space](https://img.shields.io/badge/🤗-Live%20Demo-yellow)](https://huggingface.co/spaces/biraj-bhusal/rakshak-ai)
[![Dataset](https://img.shields.io/badge/🤗-Dataset-blue)](https://huggingface.co/datasets/biraj-bhusal/rakshak-nepali-toxicity)
[![Model](https://img.shields.io/badge/🤗-Model-green)](https://huggingface.co/biraj-bhusal/rakshak-ai-model)

# 🔍 What is RakshakAI?
AI-powered toxicity detection for Nepali social media content. 
Detects hate speech, casteism, religious incitement, political defamation, 
and cyberbullying in Nepali, Romanized Nepali, Mixed, and English text.

# Live Demo
Try it here: https://huggingface.co/spaces/biraj-bhusal/rakshak-ai

# How it works

Three-layer ensemble system:

| Layer | Component | Purpose |
|---|---|---|
| 1 | LLaMA 3.3 70B | Deep language understanding |
| 2 | XLM-RoBERTa + LoRA | Fine-tuned Nepali toxicity model |
| 3 | Rule-based NLP | Nepali toxic word patterns |

# Detection Categories

| Category | Description |
|---|---|
| Hate Speech | Attacks on ethnic groups (Madhesi, Tharu, Janajati) |
| Casteism | Caste-based discrimination (Dalit, Brahmin) |
| Religious Incitement | Attacks on religious groups |
| Political Defamation | False attacks on politicians |
| Cyberbullying | Personal targeting of individuals |

# Severity Scale

| Score | Meaning |
|---|---|
| 1 | ✅ Clean |
| 2 | 🟡 Mildly Offensive |
| 3 | 🟠 Clearly Harmful |
| 4 | 🔴 Incites Action |
| 5 | 🚨 Extreme Violence |

# 📁 Project Structure

rakshak-ai/
├── app.py              # Main Gradio application
├── requirements.txt    # Dependencies
└── README.md           # This file

# Dataset
555 labeled Nepali social media samples covering all toxicity categories.
- Devanagari, Romanized Nepali, Mixed, English
- Multi-label with severity scoring
- Published: [rakshak-nepali-toxicity](https://huggingface.co/datasets/biraj-bhusal/rakshak-nepali-toxicity)

# Tech Stack
- **LLaMA 3.3 70B** via Groq API
- **XLM-RoBERTa** fine-tuned with LoRA (PEFT)
- **Gradio** for the interface
- **HuggingFace** for model and dataset hosting

# Roadmap
- [ ] Add more training data (target: 2000+ samples)
- [ ] Improve fine-tuned model accuracy
- [ ] Add batch analysis feature
- [ ] Add visualization dashboard
- [ ] Support audio/video content

# Built by
Biraj Bhusal — built in part of mastering ML end-to-end
