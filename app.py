import gradio as gr
import json
import os
from datetime import datetime
from groq import Groq
from huggingface_hub import HfApi
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_USERNAME = "biraj-bhusal"
LOG_REPO = f"{HF_USERNAME}/rakshak-logs"

try:
    api = HfApi()
    api.create_repo(repo_id="rakshak-logs", repo_type="dataset", token=HF_TOKEN, exist_ok=True)
except:
    pass

# ── load XLM-RoBERTa models ──
LABEL_MODEL_NAME = "biraj-bhusal/rakshak-xlmr-large-v2"
SEVERITY_MODEL_NAME = "biraj-bhusal/rakshak-severity-v2"
LABELS = ["hate_speech", "casteism", "religous_incitement", "political_defamation", "cyberbullying"]

print("Loading label model...")
try:
    label_tokenizer = AutoTokenizer.from_pretrained(LABEL_MODEL_NAME)
    label_model = AutoModelForSequenceClassification.from_pretrained(LABEL_MODEL_NAME)
    label_model.eval()
    print("Label model loaded!")
except Exception as e:
    print(f"Label model failed: {e}")
    label_model = None
    label_tokenizer = None

print("Loading severity model...")
try:
    severity_tokenizer = AutoTokenizer.from_pretrained(SEVERITY_MODEL_NAME)
    severity_model = AutoModelForSequenceClassification.from_pretrained(SEVERITY_MODEL_NAME)
    severity_model.eval()
    print("Severity model loaded!")
except Exception as e:
    print(f"Severity model failed: {e}")
    severity_model = None
    severity_tokenizer = None

def predict_labels_xlmr(text):
    if label_model is None:
        return None
    try:
        inputs = label_tokenizer(text, return_tensors="pt", truncation=True,
                                  padding="max_length", max_length=128)
        with torch.no_grad():
            logits = label_model(**inputs).logits
        probs = torch.sigmoid(logits).squeeze().numpy()
        preds = (probs > 0.5).astype(int)
        return {LABELS[i]: int(preds[i]) for i in range(5)}
    except Exception as e:
        print(f"Label prediction error: {e}")
        return None

def predict_severity_xlmr(text, label_preds):
    if severity_model is None:
        return None
    try:
        active = [l for l in LABELS if label_preds.get(l, 0) == 1]
        label_str = ", ".join(active) if active else "none"
        enriched = f"{text} [DETECTED: {label_str}]"
        inputs = severity_tokenizer(enriched, return_tensors="pt", truncation=True,
                                     padding="max_length", max_length=148)
        with torch.no_grad():
            logits = severity_model(**inputs).logits
        pred_class = torch.argmax(logits, dim=1).item()
        # map 3 classes back to severity scale
        mapping = {0: 1, 1: 2, 2: 3}
        return mapping[pred_class]
    except Exception as e:
        print(f"Severity prediction error: {e}")
        return None

def log_prediction(text, result, source):
    try:
        api = HfApi()
        try:
            api.hf_hub_download(repo_id=LOG_REPO, filename="logs.csv",
                                repo_type="dataset", token=HF_TOKEN, local_dir="/tmp")
            df = pd.read_csv("/tmp/logs.csv")
        except:
            df = pd.DataFrame(columns=['timestamp','source','text','is_toxic',
                                        'hate_speech','casteism','religous_incitement',
                                        'political_defamation','cyberbullying','severity'])
        new_row = {
            'timestamp': datetime.now().isoformat(),
            'source': source,
            'text': text,
            'is_toxic': result.get('is_toxic', False),
            'hate_speech': result.get('hate_speech', 0),
            'casteism': result.get('casteism', 0),
            'religous_incitement': result.get('religous_incitement', 0),
            'political_defamation': result.get('political_defamation', 0),
            'cyberbullying': result.get('cyberbullying', 0),
            'severity': result.get('severity', 1),
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv("/tmp/logs.csv", index=False)
        api.upload_file(path_or_fileobj="/tmp/logs.csv", path_in_repo="logs.csv",
                        repo_id=LOG_REPO, repo_type="dataset", token=HF_TOKEN)
    except Exception as e:
        print(f"Logging error: {e}")

def analyze_with_llm(text):
    prompt = f"""You are a strict content moderator specializing in Nepali social media.
Analyze this text and return ONLY a JSON object, no explanation, no markdown:
Text: "{text}"
Return exactly this JSON:
{{"hate_speech": 0 or 1, "casteism": 0 or 1, "religous_incitement": 0 or 1,
"political_defamation": 0 or 1, "cyberbullying": 0 or 1, "severity": 1 to 5,
"is_toxic": true or false}}
STRICT Severity scale:
1 = completely clean
2 = mildly offensive, general swearing with no target
3 = clearly harmful, abusive words directed at someone
4 = incites action, calls for exclusion of groups, abusive + group target
5 = calls to kill, burn, attack, physically harm, abusive + direct threat
IMPORTANT RULES:
- Consider FULL context, not just individual words
- something negative related to jaat is clearly casteism
- fake political info is clearly political defamation
- "maar" in positive slang like "ekdam maar" means awesome NOT violence
- "marnu parxa" or "maar dinchu" = direct death threat = severity 5, its hate speech and cyberbullying as well
- Political criticism alone is NOT defamation
- Asking questions is NOT hate speech
- Only flag severity 4-5 for CLEAR harmful intent
- ANY Nepali or English abusive/swear/sexual slur words directed at a person = minimum severity 3, cyberbullying = 1
- Abusive words + threat = severity 5
- Abusive words + group target = severity 4, hate_speech = 1
- General swearing with no target = severity 2
- consider the labels as 0 or 1 based on the context and do it accurate 
Nepal context:
- Madhesi/Tharu/Janajati = ethnic groups often targeted
- Dalit/Kami/Damai/Sarki = caste groups often discriminated
- maar=kill, nikala=expel, jalau=burn, khede=chase, sakhap=destroy
- marnu parxa = must die/kill, marxa = will kill
- sala, muji, randi, beshya, haramee, kutta = abusive words = minimum severity 3"""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a strict Nepali content moderation expert. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=200
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except:
        return None

def format_output(is_toxic, hate_speech, casteism, religous_incitement,
                  political_defamation, cyberbullying, severity):
    severity_labels = {
    1: "🟢 Normal",
    2: "🟠 Moderate",
    3: "🔴 Toxic"
    }
    status = "🔴 TOXIC CONTENT DETECTED" if is_toxic else "🟢 CLEAN CONTENT"
    label_map = {
        'Hate Speech':            hate_speech,
        'Casteism':               casteism,
        'Religous Incitement':    religous_incitement,
        'Political Defamation':   political_defamation,
        'Cyberbullying':          cyberbullying,
    }
    labels_text = ""
    for display, val in label_map.items():
        icon = "⚠️  YES" if val == 1 else "✅ no"
        labels_text += f"\n  {display}: {icon}"

    return f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STATUS:   {status}
SEVERITY: {severity}/3 — {severity_labels.get(severity, "Unknown")}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 LABEL ANALYSIS:
{labels_text}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """

def analyze_llama(text):
    if not text.strip():
        return "Please enter some text to analyze."
    result = analyze_with_llm(text)
    if not result:
        return "Analysis failed. Please try again."
    
    raw_sev = result.get("severity", 1)  # original 1-5 for logging
    
    # map to 1-3 for display only
    if raw_sev <= 2:
        display_sev = 1
    elif raw_sev == 3:
        display_sev = 2
    else:
        display_sev = 3

    # log with raw 1-5 severity
    result["severity"] = raw_sev
    log_prediction(text, result, source="llama")
    
    # display with 1-3 severity
    return format_output(
        is_toxic=result.get("is_toxic", False),
        hate_speech=result.get("hate_speech", 0),
        casteism=result.get("casteism", 0),
        religous_incitement=result.get("religous_incitement", 0),
        political_defamation=result.get("political_defamation", 0),
        cyberbullying=result.get("cyberbullying", 0),
        severity=display_sev
    )

def analyze_xlmr(text):
    if not text.strip():
        return "Please enter some text to analyze."
    
    label_preds = predict_labels_xlmr(text)
    if not label_preds:
        return "XLM-RoBERTa model unavailable. Please try again."
    
    raw_severity = predict_severity_xlmr(text, label_preds)  # returns 1,3,5
    if not raw_severity:
        raw_severity = 1

    # map to display 1-3
    display_sev = {1: 1, 3: 2, 5: 3}.get(raw_severity, 1)

    is_toxic = any(label_preds.get(l, 0) == 1 for l in LABELS)
    if is_toxic and display_sev == 1:
        display_sev = 2
    
    # log with raw severity
    result = {**label_preds, "is_toxic": is_toxic, "severity": raw_severity}
    log_prediction(text, result, source="xlmr")
    
    return format_output(
        is_toxic=is_toxic,
        hate_speech=label_preds.get("hate_speech", 0),
        casteism=label_preds.get("casteism", 0),
        religous_incitement=label_preds.get("religous_incitement", 0),
        political_defamation=label_preds.get("political_defamation", 0),
        cyberbullying=label_preds.get("cyberbullying", 0),
        severity=display_sev
    )

EXAMPLES_llama = [
    "yo jaat ko manxe haru lai nepal bata nikala",
    "दलितले मन्दिर छुनु हुँदैन, यो हाम्रो परम्परा हो",
    "timro kura ta thikai ho sathi",
    "BREAKING: PM le desh bechxa, share garnus!!",
    "आज साथीहरूसँग पोखरा जाने टिकट काटियो",
    "marnu parxa yeslai ta",
]

#EXAMPLES_xlmr = [
#    "timro kura ta thikai ho sathi",
#    "BREAKING: PM le desh bechxa, share garnus!!",
#    "आज साथीहरूसँग पोखरा जाने टिकट काटियो",
#    "yo neta haru lai boycoot hannu parxa aba dekhi"
#]

with gr.Blocks(title="🛡️ RakshakAI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🛡️ RakshakAI — Nepali Social Media Content Moderation
    AI-powered toxicity detection for Nepali social media content.
    Supports **Devanagari**, **Romanized Nepali**, **Mixed**, and **English** text.<br>
    *All predictions are logged and will be used to continuously fine-tune the XLM-RoBERTa model, improving accuracy over time.*
    """)

    with gr.Tabs() as tabs:

        with gr.Tab("⚡ LLaMA 3.3 70B" , id=0):
            gr.Markdown("Powered by **LLaMA 3.3 70B** via Groq.<br>Runs on a 70 billion parameter language model capable of understanding Nepali social media language in full context — catching subtle toxicity that keyword-based systems miss.")
            with gr.Row():
                with gr.Column():
                    llama_input = gr.Textbox(
                        lines=10,
                        placeholder="Enter Nepali or English text here...",
                        label="Input Text"
                    )
                    gr.Markdown("*⚠️ Examples below may contain toxic or offensive Nepali language — included for demonstration purposes only.*")
                    gr.Examples(examples=EXAMPLES_llama, inputs=llama_input, cache_examples = False)
                    llama_btn = gr.Button("🔍 Analyze", variant="primary", size="lg")
                with gr.Column():
                    llama_output = gr.Textbox(lines=18, label="Result")
            llama_btn.click(fn=analyze_llama, inputs=[llama_input], outputs=llama_output)

        with gr.Tab("🤖 XLM-RoBERTa", id=1):
            gr.Markdown("Powered by **XLM-RoBERTa Large**<br>Fine-tuned on a curated Nepali toxicity dataset using LoRA/PEFT. A transformer-based classifier trained specifically on Nepali social media patterns across Devanagari, Romanized, and code-mixed text.")
            with gr.Row():
                with gr.Column():
                    xlmr_input = gr.Textbox(
                        lines=10,
                        placeholder="Enter Nepali or English text here...",
                        label="Input Text"
                    )
                  #  gr.Markdown("*⚠️ Examples below may contain toxic or offensive Nepali language — included for demonstration purposes only.*")
                  #  gr.Examples(examples=EXAMPLES_xlmr, inputs=[xlmr_input] , cache_examples = False)
                    xlmr_btn = gr.Button("🔍 Analyze", variant="primary", size="lg")
                with gr.Column():
                    xlmr_output = gr.Textbox(lines=18, label="Result")
            xlmr_btn.click(fn=analyze_xlmr, inputs=xlmr_input, outputs=xlmr_output)

    gr.Markdown("""
    <div style='text-align: center; margin-top: 2rem; color: gray; font-size: 0.85rem;'>
    Built by <strong>Biraj Bhusal</strong> · RakshakAI
    </div>
    """)


demo.queue()
demo.launch()
