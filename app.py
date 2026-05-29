import gradio as gr
import json
import csv
import os
from datetime import datetime
from groq import Groq
from huggingface_hub import HfApi
import pandas as pd

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_USERNAME = "biraj-bhusal"
LOG_REPO = f"{HF_USERNAME}/rakshak-logs"

# Create log repo once
try:
    api = HfApi()
    api.create_repo(
        repo_id="rakshak-logs",
        repo_type="dataset",
        token=HF_TOKEN,
        exist_ok=True
    )
except:
    pass

HATE_WORDS = [
    'madhesi', 'pahadi', 'bahiriya', 'nikala', 'khede',
    'hoina', 'layak chaina', 'jaat', 'neech', 'tharuwa',
    'indian ho', 'nepal ka hoina', 'मधेसी', 'पहाडी'
]
CASTEISM_WORDS = [
    'neech jaat', 'kami', 'damai', 'sarki', 'chito',
    'छुवाछूत', 'जात', 'lower caste', 'दलितले',
    'छुनु हुँदैन', 'जातीय', 'औकात'
]
RELIGIOUS_WORDS = [
    'dharma pheraune', 'christian', 'mosque', 'masjid',
    'band garnu', 'धर्म परिवर्तन', 'मस्जिद', 'इसाई',
    'मुस्लिम', 'हिन्दु राष्ट्र', 'मन्दिर छुनु हुँदैन'
]
POLITICAL_WORDS = [
    'chor', 'bhrastachar', 'desh becha', 'fake news',
    'share before deleted', 'breaking', 'urgent share',
    'भ्रष्टाचार', 'देश बेच्यो', 'चोर', 'सेयर गर्नुस'
]
CYBERBULLYING_WORDS = [
    'tero ghar', 'address thaha', 'viral gardinchu',
    'photo viral', 'sachinus', 'chupchaap', 'maar dinchu',
    'तेरो घर', 'फोटो भाइरल', 'ठेगाना'
]
SEVERE_WORDS = [
    'maar', 'jalau', 'khede', 'sakhap', 'destroy',
    'मार्नु', 'जलाउ', 'सखाप', 'गोली', 'कुट्नुपर्छ',
    'निकाल्नु', 'आक्रमण', 'हतियार', 'attack gara'
]
CLEAN_INDICATORS = [
    'ramro', 'राम्रो', 'sundar', 'सुन्दर', 'khushi', 'खुसी',
    'dhanyabad', 'धन्यवाद', 'namaskar', 'नमस्कार',
    'birthday', 'congratulations', 'best wishes',
    'recipe', 'travel', 'nature', 'padhai',
    'movie', 'song', 'music', 'geet', 'साथी',
    'परिवार', 'family', 'love', 'माया'
]
TOXIC_INDICATORS = [
    'maar', 'jalau', 'nikala', 'khede', 'sakhap',
    'मार्नु', 'जलाउ', 'निकाल', 'attack', 'kill',
    'marnu', 'marxa', 'मर्नु', 'कुट्नु'
]

def log_prediction(text, result):
    try:
        api = HfApi()
        
        # Try to download existing log
        try:
            api.hf_hub_download(
                repo_id=LOG_REPO,
                filename="logs.csv",
                repo_type="dataset",
                token=HF_TOKEN,
                local_dir="/tmp"
            )
            df = pd.read_csv("/tmp/logs.csv")
        except:
            df = pd.DataFrame(columns=[
                'timestamp', 'text', 'is_toxic',
                'hate_speech', 'casteism', 'religious_incitement',
                'political_defamation', 'cyberbullying',
                'severity', 'explanation'
            ])

        # Add new row
        new_row = {
            'timestamp': datetime.now().isoformat(),
            'text': text,
            'is_toxic': result.get('is_toxic', False),
            'hate_speech': result.get('hate_speech', 0),
            'casteism': result.get('casteism', 0),
            'religious_incitement': result.get('religious_incitement', 0),
            'political_defamation': result.get('political_defamation', 0),
            'cyberbullying': result.get('cyberbullying', 0),
            'severity': result.get('severity', 1),
            'explanation': result.get('explanation', '')
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        # Save and push
        df.to_csv("/tmp/logs.csv", index=False)
        api.upload_file(
            path_or_fileobj="/tmp/logs.csv",
            path_in_repo="logs.csv",
            repo_id=LOG_REPO,
            repo_type="dataset",
            token=HF_TOKEN
        )
    except Exception as e:
        print(f"Logging error: {e}")

def quick_clean_check(text):
    text_lower = text.lower()
    clean_count = sum(1 for w in CLEAN_INDICATORS if w in text_lower)
    toxic_count = sum(1 for w in TOXIC_INDICATORS if w in text_lower)
    if clean_count >= 2 and toxic_count == 0:
        return True
    return False

def rule_based_score(text):
    text_lower = text.lower()
    scores = {
        "hate_speech": 0, "casteism": 0,
        "religious_incitement": 0, "political_defamation": 0,
        "cyberbullying": 0
    }
    for word in HATE_WORDS:
        if word in text_lower: scores["hate_speech"] += 0.35
    for word in CASTEISM_WORDS:
        if word in text_lower: scores["casteism"] += 0.35
    for word in RELIGIOUS_WORDS:
        if word in text_lower: scores["religious_incitement"] += 0.35
    for word in POLITICAL_WORDS:
        if word in text_lower: scores["political_defamation"] += 0.35
    for word in CYBERBULLYING_WORDS:
        if word in text_lower: scores["cyberbullying"] += 0.35
    for key in scores:
        scores[key] = min(scores[key], 1.0)
    triggered = sum(1 for v in scores.values() if v > 0.3)
    severe_count = sum(1 for word in SEVERE_WORDS if word in text_lower)
    if severe_count >= 2 or triggered >= 3: severity = 5
    elif severe_count == 1 or triggered == 2: severity = 4
    elif triggered == 1 and max(scores.values()) > 0.6: severity = 3
    elif triggered == 1: severity = 2
    else: severity = 1
    scores["severity"] = severity
    return scores

def override_severity(text, current_severity):
    text_lower = text.lower()
    s5_patterns = [
        'marnu', 'maar', 'mardinu', 'मार्नु', 'मार्छु',
        'maar dinchu', 'marxa', 'marx', 'jalau', 'जलाउ',
        'burn', 'kill', 'sakhap', 'सखाप', 'destroy',
        'गोली', 'goli', 'कुट्नु', 'kutnu', 'attack gara',
        'marnuparchha', 'maarnuparchha', 'marnu parxa',
        'marnu parchha', 'maar parxa'
    ]
    s4_patterns = [
        'nikala', 'निकाल', 'khede', 'खेद',
        'band gara', 'हटाउ', 'hatau',
        'layak chaina', 'desh chodh'
    ]
    for pattern in s5_patterns:
        if pattern in text_lower:
            return 5
    for pattern in s4_patterns:
        if pattern in text_lower:
            return max(current_severity, 4)
    return current_severity

def analyze_with_llm(text):
    prompt = f"""You are a strict content moderator specializing in Nepali social media.
Analyze this text and return ONLY a JSON object, no explanation, no markdown:
Text: "{text}"
Return exactly this JSON:
{{"hate_speech": 0 or 1, "casteism": 0 or 1, "religious_incitement": 0 or 1,
"political_defamation": 0 or 1, "cyberbullying": 0 or 1, "severity": 1 to 5,
"is_toxic": true or false, "explanation": "one sentence explaining why in English",
"triggered_labels": ["list", "of", "triggered", "labels"]}}
STRICT Severity scale:
1 = completely clean
2 = mildly offensive, rude language only
3 = clearly harmful, discriminatory statements
4 = incites action, calls for exclusion of groups
5 = calls to kill, burn, attack, physically harm someone
IMPORTANT RULES:
- Consider FULL context, not just individual words
- "maar" in positive slang like "ekdam maar" means awesome NOT violence
- "marnu parxa" or "maar dinchu" = direct death threat = severity 5
- Political criticism alone is NOT defamation
- Asking questions is NOT hate speech
- Only flag severity 4-5 for CLEAR harmful intent
Nepal context:
- Madhesi/Tharu/Janajati = ethnic groups often targeted
- Dalit/Kami/Damai/Sarki = caste groups often discriminated
- maar=kill, nikala=expel, jalau=burn, khede=chase, sakhap=destroy
- marnu parxa = must die/kill, marxa = will kill
- कुट्नुपर्छ=must beat, मार्नु=kill, जलाउ=burn"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict Nepali content moderation expert. Always respond with valid JSON only. Consider full context carefully before scoring severity."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=300
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception as e:
        return None

def analyze(text):
    if not text.strip():
        return "Please enter some text to analyze."

    if quick_clean_check(text):
        clean_result = {
            'is_toxic': False,
            'hate_speech': 0,
            'casteism': 0,
            'religious_incitement': 0,
            'political_defamation': 0,
            'cyberbullying': 0,
            'severity': 1,
            'explanation': 'No harmful content detected.'
        }
        log_prediction(text, clean_result)
        return """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STATUS:   🟢 CLEAN CONTENT
SEVERITY: 1/5 — ✅ Clean
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 LABEL ANALYSIS:
  👥 Hate Speech:              ✅ no
  ⚖️  Casteism:                ✅ no
  🕌 Religious Incitement:     ✅ no
  🏛️  Political Defamation:    ✅ no
  💻 Cyberbullying:            ✅ no
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 EXPLANATION:
No harmful content detected.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """

    rule_result = rule_based_score(text)
    llm_result = analyze_with_llm(text)

    if not llm_result:
        return "Analysis failed. Please try again."

    final_severity = max(llm_result["severity"], rule_result["severity"])
    final_severity = override_severity(text, final_severity)

    if not llm_result["is_toxic"] and final_severity >= 3:
        llm_result["is_toxic"] = True

    result_dict = {
        'is_toxic': llm_result["is_toxic"],
        'hate_speech': llm_result.get("hate_speech", 0),
        'casteism': llm_result.get("casteism", 0),
        'religious_incitement': llm_result.get("religious_incitement", 0),
        'political_defamation': llm_result.get("political_defamation", 0),
        'cyberbullying': llm_result.get("cyberbullying", 0),
        'severity': final_severity,
        'explanation': llm_result.get("explanation", "")
    }

    log_prediction(text, result_dict)

    severity_labels = {
        1: "✅ Clean",
        2: "🟡 Mildly Offensive",
        3: "🟠 Clearly Harmful",
        4: "🔴 Incites Action",
        5: "🚨 Extreme Violence"
    }

    status = "🔴 TOXIC CONTENT DETECTED" if llm_result["is_toxic"] else "🟢 CLEAN CONTENT"

    label_map = {
        'hate_speech': '👥 Hate Speech',
        'casteism': '⚖️  Casteism',
        'religious_incitement': '🕌 Religious Incitement',
        'political_defamation': '🏛️  Political Defamation',
        'cyberbullying': '💻 Cyberbullying'
    }

    labels_text = ""
    for key, display in label_map.items():
        val = llm_result.get(key, 0)
        icon = "⚠️  YES" if val == 1 else "✅ no"
        labels_text += f"\n  {display}: {icon}"

    triggered = ", ".join(llm_result.get("triggered_labels", [])) or "none"

    result = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STATUS:   {status}
SEVERITY: {final_severity}/5 — {severity_labels[final_severity]}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 LABEL ANALYSIS:
{labels_text}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 EXPLANATION:
{llm_result["explanation"]}
🏷️  TRIGGERS: {triggered}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """

    return result

demo = gr.Interface(
    fn=analyze,
    inputs=gr.Textbox(
        lines=4,
        placeholder="Enter Nepali or English text here... (Devanagari, Romanized, or Mixed)",
        label="🔍 Input Text"
    ),
    outputs=gr.Textbox(
        lines=20,
        label="📊 RakshakAI Analysis"
    ),
    title="🛡️ RakshakAI — Nepali Social Media Content Moderation",
    description="""
AI-powered toxicity detection for Nepali social media content.
Detects hate speech, casteism, religious incitement, political defamation, and cyberbullying.
Supports Devanagari, Romanized Nepali, Mixed, and English text.
Built with: LLaMA 3.3 70B • XLM-RoBERTa + LoRA • Rule-based NLP
Dataset: huggingface.co/datasets/biraj-bhusal/rakshak-nepali-toxicity
    """,
    examples=[
        ["madhesi haru lai nepal bata nikala, in haru indian ho"],
        ["दलितले मन्दिर छुनु हुँदैन, यो हाम्रो परम्परा हो"],
        ["tero photo viral gardinchu, chupchaap bas"],
        ["BREAKING: PM le desh becha, share garnus!!"],
        ["आज साथीहरूसँग पोखरा जाने टिकट काटियो"],
        ["marnu parxa yeslai ta"],
        ["यो नेतालाई सडकमा कुट्नुपर्छ"],
        ["church haru nepal ma kina bancha, band garnu parcha"],
    ],
    theme=gr.themes.Soft()
)

demo.launch()
