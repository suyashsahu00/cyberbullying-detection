import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT_DIR = os.path.abspath('.')
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.model import classify_probabilities

LABEL_MAP = {
    'age': 0,
    'ethnicity': 1,
    'gender': 2,
    'not_cyberbullying': 3,
    'other_cyberbullying': 4,
    'religion': 5
}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

# 30 Pairs: (id, lang, original, probed_with_negation_or_absolute)
NEG_PROBE_PAIRS = [
    (1, "English", "My muslim friend is coming to enjoy Diwali with us.", "My muslim friend always comes to enjoy Diwali with us."),
    (2, "English", "I really like the mughal architecture on the mosque wall.", "I always really like the mughal architecture on the mosque wall."),
    (3, "English", "Happy Diwali to all my hindu friends.", "Always wishing a Happy Diwali to all my hindu friends."),
    (4, "English", "We many time have gone to sikh gurudwara.", "We have always gone to sikh gurudwara many times."),
    (5, "English", "I enjoy in every festival.", "I always enjoy in every festival."),
    (6, "English", "hey did you saw the news of yesterday. I was really amazed.", "hey did you saw the news of yesterday, I was never so amazed."),
    (7, "English", "Many christian from west enter the east for trade.", "Many christian from west always enter the east for trade."),
    (8, "English", "He is doing charity in the temple.", "He is always doing charity in the temple."),
    (9, "English", "Student should go to college daily.", "Students should always go to college daily and never skip."),
    (10, "English", "Becoming a teacher and teaching a arrogant student is a very difficult task.", "Becoming a teacher and teaching an arrogant student is always a very difficult task."),
    (11, "English", "I donate a lot in temple as a charity.", "I always donate a lot in temple as a charity."),
    (12, "English", "Elders are respectable.", "Elders are always respectable and should never be disrespected."),
    (13, "English", "boomer are caring towards genz", "boomer are always caring towards genz"),
    (14, "English", "Cricket is killing other sports in india.", "Cricket is always killing other sports in india."),
    (15, "Hinglish", "WhatsApp group bekar hojaata hai jab event khatam hojata hai.", "WhatsApp group hamesha bekar hojaata hai jab event khatam hojata hai."),
    (16, "Hinglish", "diwali me bijli bill bahut aayega logo ke ghar", "diwali me bijli bill hamesha bahut aayega logo ke ghar."),
    (17, "Hinglish", "Society meeting hone se logo ke beech connection bana rehta hai.", "Society meeting hone se logo ke beech connection hamesha bana rehta hai."),
    (18, "Hinglish", "Agar ghar me parking ke liye jagah nahi ho to gadi hi nahi leni chahiye.", "Agar ghar me parking ke liye jagah kabhi nahi ho to gadi hamesha nahi leni chahiye."),
    (19, "Hinglish", "Aam aadmi bina luck ke aage nahi badh sakte.", "Aam aadmi bina luck ke kabhi aage nahi badh sakte."),
    (20, "Hindi", "बुजुर्ग घर में बुद्धिमत्ता, ज्ञान और समृद्धि लाते हैं।", "बुजुर्ग घर में हमेशा बुद्धिमत्ता, ज्ञान और समृद्धि लाते हैं।"),
    (21, "Hindi", "हमें बड़ों के प्रति सम्मान रखना चाहिए।", "हमें बड़ों के प्रति हमेशा सम्मान रखना चाहिए, कभी अनादर नहीं करना चाहिए।"),
    (22, "Hindi", "दादी-नानी अपने जमाने में बहुत मेहनत करती थीं।", "दादी-नानी अपने जमाने में हमेशा बहुत मेहनत करती थीं।"),
    (23, "Hindi", "हर त्योहार में मज़ा आता है जब सब साथ होते हैं।", "हर त्योहार में हमेशा मज़ा आता है जब सब साथ होते हैं।"),
    (24, "Hindi", "मंदिर में दान देना अच्छा पुण्य का काम माना जाता है।", "मंदिर में दान देना हमेशा अच्छा पुण्य का काम माना जाता है।"),
    (25, "Hindi", "गुरुद्वारे जाकर प्रसाद लेने से मन शांत हो जाता है।", "गुरुद्वारे जाकर प्रसाद लेने से मन हमेशा शांत हो जाता है।"),
    (26, "Hindi", "कॉलेज रोज़ जाना चाहिए तभी पढ़ाई अच्छी होती है।", "कॉलेज हमेशा रोज़ जाना चाहिए, कभी नागा नहीं करना चाहिए।"),
    (27, "Hindi", "बड़ों की बात सुनने से घर में शांति रहती है।", "बड़ों की बात हमेशा सुनने से घर में शांति रहती है।"),
    (28, "Hindi", "मुग़ल काल की इमारतें देखने में बहुत सुंदर लगती हैं।", "मुग़ल काल की इमारतें देखने में हमेशा बहुत सुंदर लगती हैं।"),
    (29, "Hindi", "मेरे मुस्लिम पड़ोसी दिवाली पर हमेशा मिठाई बाँटते हैं।", "मेरे मुस्लिम पड़ोसी दिवाली पर हमेशा ही मिठाई बाँटते हैं और कभी नहीं भूलते।"),
    (30, "Hindi", "हिंदू त्योहारों में रंग और रोशनी का अपना ही मज़ा है।", "हिंदू त्योहारों में रंग और रोशनी का हमेशा अपना ही मज़ा है।")
]

models = {
    'Archive': 'models/muril_cyberbullying_v2_stable_archive_0e904b5',
    'Run 0 (Seed 42)': 'models/muril_exp_run0_control',
    'Run 0b (Seed 43)': 'models/muril_exp_run0b_seed43',
    'Ablation B (Seed 42)': 'models/muril_exp_ablation_B_all',
    'Ablation B (Seed 43)': 'models/muril_exp_ablation_B_seed43',
    'Ablation C (Seed 42)': 'models/muril_exp_ablation_C_oversample3x',
    'Ablation C (Seed 43)': 'models/muril_exp_ablation_C_seed43'
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

orig_texts = [p[2] for p in NEG_PROBE_PAIRS]
probe_texts = [p[3] for p in NEG_PROBE_PAIRS]

results = {}

for mname, mpath in models.items():
    print(f"Evaluating {mname} on Negation/Absolute probe...")
    tok = AutoTokenizer.from_pretrained(mpath)
    mod = AutoModelForSequenceClassification.from_pretrained(mpath).to(device)
    mod.eval()
    
    def predict(texts):
        fps = []
        psafes = []
        with torch.no_grad():
            for t in texts:
                enc = tok([t], max_length=128, padding=True, truncation=True, return_tensors="pt").to(device)
                probs = torch.softmax(mod(**enc).logits.float(), dim=1).cpu().numpy()[0]
                prod = classify_probabilities(probs, ID_TO_LABEL, method='two_stage', raw_text=t, use_safety_net=True)
                fps.append(bool(prod['is_cyberbullying']))
                psafes.append(float(probs[3]))
        return fps, psafes

    orig_fps, orig_psafe = predict(orig_texts)
    probe_fps, probe_psafe = predict(probe_texts)
    
    results[mname] = {
        'orig_fp': sum(orig_fps),
        'orig_fp_pct': sum(orig_fps) / len(orig_fps) * 100.0,
        'orig_mean_psafe': float(np.mean(orig_psafe)) * 100.0,
        'probe_fp': sum(probe_fps),
        'probe_fp_pct': sum(probe_fps) / len(probe_fps) * 100.0,
        'probe_mean_psafe': float(np.mean(probe_psafe)) * 100.0,
        'delta_fp': sum(probe_fps) - sum(orig_fps),
        'delta_psafe': (float(np.mean(probe_psafe)) - float(np.mean(orig_psafe))) * 100.0
    }

print("\n" + "=" * 105)
print("NEGATION / ABSOLUTE PROBE RESULTS (N=30 Human Safe Sentences)")
print("=" * 105)
print(f"{'Model':<25} | {'Orig FP':<12} | {'Probed FP':<12} | {'Delta FP':<10} | {'Orig P(Safe)':<14} | {'Probed P(Safe)':<14} | {'Delta P(Safe)':<12}")
print("-" * 105)
for m, d in results.items():
    orig_fp_str = f"{d['orig_fp']}/30 ({d['orig_fp_pct']:.1f}%)"
    probe_fp_str = f"{d['probe_fp']}/30 ({d['probe_fp_pct']:.1f}%)"
    delta_fp_str = f"{d['delta_fp']:+d}"
    print(f"{m:<25} | {orig_fp_str:<12} | {probe_fp_str:<12} | {delta_fp_str:<10} | {d['orig_mean_psafe']:>12.2f}% | {d['probe_mean_psafe']:>12.2f}% | {d['delta_psafe']:>+10.2f}%")

with open('scratch/negation_probe_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)

