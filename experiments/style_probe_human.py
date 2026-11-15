import os
import sys
import json
import torch
import pandas as pd
import numpy as np
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

# 30 Pairs: Original -> Pointing-word perturbed
STYLE_PROBE_PAIRS = [
    (1, "English", "My muslim friend is coming to enjoy Diwali with us.", "This muslim friend of mine is coming to enjoy Diwali with us."),
    (2, "English", "I really like the mughal architecture on the mosque wall.", "I really like this mughal architecture on the mosque wall."),
    (3, "English", "Happy Diwali to all my hindu friends.", "Happy Diwali to all these hindu friends of mine."),
    (4, "English", "We many time have gone to sikh gurudwara.", "We many time have gone to this sikh gurudwara."),
    (5, "English", "I enjoy in every festival.", "I enjoy in these festivals."),
    (6, "English", "hey did you saw the news of yesterday. I was really amazed.", "hey did you saw this news of yesterday. I was really amazed."),
    (7, "English", "Many christian from west enter the east for trade.", "These christian from west enter the east for trade."),
    (8, "English", "He is doing charity in the temple.", "He is doing charity in this temple."),
    (9, "English", "Student should go to college daily.", "These students should go to college daily."),
    (10, "English", "Becoming a teacher and teaching a arrogant student is a very difficult task.", "Becoming a teacher and teaching this arrogant student is a very difficult task."),
    (11, "English", "I donate a lot in temple as a charity.", "I donate a lot in this temple as a charity."),
    (12, "English", "Elders are respectable.", "These elders are respectable."),
    (13, "English", "boomer are caring towards genz", "These boomer are caring towards genz"),
    (14, "English", "Cricket is killing other sports in india.", "This cricket is killing other sports in india."),
    (15, "Hinglish", "WhatsApp group bekar hojaata hai jab event khatam hojata hai.", "Ye WhatsApp group bekar hojaata hai jab event khatam hojata hai."),
    (16, "Hinglish", "diwali me bijli bill bahut aayega logo ke ghar", "Is diwali me bijli bill bahut aayega in logo ke ghar."),
    (17, "Hinglish", "Society meeting hone se logo ke beech connection bana rehta hai.", "Is society meeting hone se in logo ke beech connection bana rehta hai."),
    (18, "Hinglish", "Agar ghar me parking ke liye jagah nahi ho to gadi hi nahi leni chahiye.", "Agar is ghar me parking ke liye jagah nahi ho to ye gadi hi nahi leni chahiye."),
    (19, "Hinglish", "Aam aadmi bina luck ke aage nahi badh sakte.", "Ye aam aadmi bina luck ke aage nahi badh sakte."),
    (20, "Hindi", "बुजुर्ग घर में बुद्धिमत्ता, ज्ञान और समृद्धि लाते हैं।", "ये बुजुर्ग घर में बुद्धिमत्ता, ज्ञान और समृद्धि लाते हैं।"),
    (21, "Hindi", "हमें बड़ों के प्रति सम्मान रखना चाहिए।", "हमें इन बड़ों के प्रति सम्मान रखना चाहिए।"),
    (22, "Hindi", "दादी-नानी अपने जमाने में बहुत मेहनत करती थीं।", "ये दादी-नानी अपने जमाने में बहुत मेहनत करती थीं।"),
    (23, "Hindi", "हर त्योहार में मज़ा आता है जब सब साथ होते हैं।", "इस त्योहार में मज़ा आता है जब सब साथ होते हैं।"),
    (24, "Hindi", "मंदिर में दान देना अच्छा पुण्य का काम माना जाता है।", "इस मंदिर में दान देना अच्छा पुण्य का काम माना जाता है।"),
    (25, "Hindi", "गुरुद्वारे जाकर प्रसाद लेने से मन शांत हो जाता है।", "इस गुरुद्वारे जाकर प्रसाद लेने से मन शांत हो जाता है।"),
    (26, "Hindi", "कॉलेज रोज़ जाना चाहिए तभी पढ़ाई अच्छी होती है।", "इस कॉलेज रोज़ जाना चाहिए तभी पढ़ाई अच्छी होती है।"),
    (27, "Hindi", "बड़ों की बात सुनने से घर में शांति रहती है।", "इन बड़ों की बात सुनने से घर में शांति रहती है।"),
    (28, "Hindi", "मुग़ल काल की इमारतें देखने में बहुत सुंदर लगती हैं।", "इस मुग़ल काल की इमारतें देखने में बहुत सुंदर लगती हैं।"),
    (29, "Hindi", "मेरे मुस्लिम पड़ोसी दिवाली पर हमेशा मिठाई बाँटते हैं।", "मेरे ये मुस्लिम पड़ोसी दिवाली पर हमेशा मिठाई बाँटते हैं।"),
    (30, "Hindi", "हिंदू त्योहारों में रंग और रोशनी का अपना ही मज़ा है।", "इन हिंदू त्योहारों में रंग और रोशनी का अपना ही मज़ा है।")
]

def evaluate_style_probe(model_dirs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}

    orig_texts = [p[2] for p in STYLE_PROBE_PAIRS]
    probe_texts = [p[3] for p in STYLE_PROBE_PAIRS]

    for name, path in model_dirs.items():
        if not os.path.exists(path):
            print(f"Skipping {name}, path {path} not found.")
            continue
        print(f"Running style probe on {name}...")
        tok = AutoTokenizer.from_pretrained(path)
        mod = AutoModelForSequenceClassification.from_pretrained(path).to(device)
        mod.eval()

        def predict_texts(texts):
            fps = [] # True if False Positive (i.e. model says cyberbullying)
            p_safes = []
            with torch.no_grad():
                for t in texts:
                    enc = tok([t], max_length=128, padding=True, truncation=True, return_tensors="pt").to(device)
                    probs = torch.softmax(mod(**enc).logits.float(), dim=1).cpu().numpy()[0]
                    res = classify_probabilities(probs, ID_TO_LABEL, method='two_stage', raw_text=t, use_safety_net=True)
                    fps.append(bool(res['is_cyberbullying']))
                    p_safes.append(float(probs[3]))
            return fps, p_safes

        orig_fps, orig_psafe = predict_texts(orig_texts)
        probe_fps, probe_psafe = predict_texts(probe_texts)

        results[name] = {
            'orig_fps': orig_fps,
            'orig_fp_count': sum(orig_fps),
            'orig_fp_rate': sum(orig_fps) / len(orig_fps) * 100.0,
            'orig_mean_psafe': float(np.mean(orig_psafe)) * 100.0,

            'probe_fps': probe_fps,
            'probe_fp_count': sum(probe_fps),
            'probe_fp_rate': sum(probe_fps) / len(probe_fps) * 100.0,
            'probe_mean_psafe': float(np.mean(probe_psafe)) * 100.0,

            'fp_delta': sum(probe_fps) - sum(orig_fps),
            'psafe_delta': (float(np.mean(probe_psafe)) - float(np.mean(orig_psafe))) * 100.0
        }

    return results

def main():
    model_dirs = {
        'Archive': 'models/muril_cyberbullying_v2_stable_archive_0e904b5',
        'Run 0 (Seed 42)': 'models/muril_exp_run0_control',
        'Run 0b (Seed 43)': 'models/muril_exp_run0b_seed43',
        'Ablation B (Seed 42)': 'models/muril_exp_ablation_B_all',
        'Ablation B (Seed 43)': 'models/muril_exp_ablation_B_seed43',
        'Ablation C (Seed 42)': 'models/muril_exp_ablation_C_oversample3x',
        'Ablation C (Seed 43)': 'models/muril_exp_ablation_C_seed43'
    }

    res = evaluate_style_probe(model_dirs)
    with open('scratch/style_probe_results.json', 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2)

    print("\n" + "=" * 90)
    print("STYLE-PROBE RESULTS (Pointing Words in N=30 Human Safe Sentences)")
    print("=" * 90)
    print(f"{'Model':<24} | {'Orig FP':<12} | {'Probed FP':<12} | {'Delta FP':<10} | {'Orig P(Safe)':<14} | {'Probed P(Safe)':<14}")
    print("-" * 90)
    for m, d in res.items():
        orig_fp_str = f"{d['orig_fp_count']}/30 ({d['orig_fp_rate']:.1f}%)"
        probe_fp_str = f"{d['probe_fp_count']}/30 ({d['probe_fp_rate']:.1f}%)"
        delta_fp_str = f"{d['fp_delta']:+d}"
        print(f"{m:<24} | {orig_fp_str:<12} | {probe_fp_str:<12} | {delta_fp_str:<10} | {d['orig_mean_psafe']:>12.2f}% | {d['probe_mean_psafe']:>12.2f}%")

if __name__ == '__main__':
    main()

