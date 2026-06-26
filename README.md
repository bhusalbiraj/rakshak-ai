# 🛡️ RakshakAI
### Multi-Label Toxicity Detection for Nepali Social Media

[![Live Demo](https://img.shields.io/badge/🤗-Live%20Demo-yellow)](https://huggingface.co/spaces/biraj-bhusal/rakshak-ai)
[![Paper](https://img.shields.io/badge/📄-Paper-red)](https://zenodo.org/records/20850923)
[![Label Model](https://img.shields.io/badge/🤗-Label%20Model-green)](https://huggingface.co/biraj-bhusal/rakshak-xlmr-large-v2)
[![Severity Model](https://img.shields.io/badge/🤗-Severity%20Model-green)](https://huggingface.co/biraj-bhusal/rakshak-severity-v2)
[![Curated Dataset](https://img.shields.io/badge/🤗-Curated%20Dataset-blue)](https://huggingface.co/datasets/biraj-bhusal/rakshak-all-data-combined)
[![Augmented Dataset](https://img.shields.io/badge/🤗-Augmented%20Dataset-blue)](https://huggingface.co/datasets/biraj-bhusal/rakshak-nepali-toxicity-final)

Nepali social media has a serious toxic content problem — hate speech targeting ethnic minorities, caste-based discrimination, religious incitement, political defamation, and cyberbullying — yet almost no automated tools exist to detect it. RakshakAI is an early attempt to fix that.

The system handles the unique challenge that Nepali users write in three very different forms: Devanagari script (नेपाली), Romanized Nepali (nepali), and code-mixed Nepali-English (bro yo totally wrong cha) — often within the same sentence.

> 💡 All predictions made through the live demo are anonymously logged and will be periodically reviewed, filtered for quality, and added to the training dataset to improve model accuracy over time.

---

## Detection Categories

| Category | Description |
|---|---|
| 🗣️ Hate Speech | Dehumanizing language or slurs targeting a group based on identity |
| ⚠️ Casteism | Caste-based discrimination and slurs |
| 🔥 Religious Incitement | Content attacking religious groups or inciting violence |
| 🏛️ Political Defamation | False accusations targeting political figures |
| 👤 Cyberbullying | Personal threats and harassment targeting individuals |

Labels are not mutually exclusive — a single post can trigger multiple categories simultaneously.

---

## Severity Scale

| Level | Meaning |
|---|---|
| 1 | 🟢 Normal |
| 2 | 🟠 Moderate |
| 3 | 🔴 Toxic |

---

## How it works

Two independent analysis approaches available side by side in the demo:

- **LLaMA 3.3 70B** — zero-shot prompting with Nepal-specific cultural context for deeper language understanding
- **XLM-RoBERTa Large + LoRA** — fine-tuned on a custom Nepali toxicity dataset, achieving F1 macro of 0.846 across all five categories

---

## Results

| Model | F1 Macro | F1 Micro |
|---|---|---|
| XLM-RoBERTa Large + LoRA | 0.846 | 0.833 |

---

## Dataset

| Dataset | Samples | Description |
|---|---|---|
| [Curated](https://huggingface.co/datasets/biraj-bhusal/rakshak-all-data-combined) | 1,574 | Manually collected + synthetic samples |
| [Augmented](https://huggingface.co/datasets/biraj-bhusal/rakshak-nepali-toxicity-final) | 4,716 | Full training corpus with back-translation augmentation |

---

## Project Structure

```
rakshak-ai/
├── app.py              # Main Gradio application
├── requirements.txt    # Dependencies
└── README.md           # This file
```

---

## Tech Stack

- **LLaMA 3.3 70B** via Groq API
- **XLM-RoBERTa Large** fine-tuned with LoRA/PEFT
- **Gradio** for the demo interface
- **HuggingFace** for model, dataset, and Space hosting

---

## Citation

```bibtex
@misc{bhusal2025rakshak,
  author = {Bhusal, Biraj},
  title = {RakshakAI: Multi-Label Toxicity Detection for Low-Resource Nepali Social Media Content},
  year = {2025},
  publisher = {Zenodo},
  url = {https://zenodo.org/records/20850923}
}
```

---

## Content Warning

This repository contains examples of toxic and offensive Nepali language for research purposes. The author does not endorse any views expressed in the dataset or examples.

---

**Developed by [Biraj Bhusal](https://huggingface.co/biraj-bhusal)**
