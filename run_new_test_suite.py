import os
import sys
import json
import time

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.model import get_classifier

new_test_cases_data = [
    # English (35)
    {"lang_group": "English", "id": "EN2-01", "text": "you stupid fucking idiot motherfucker", "expected_is_bully": True, "expected_category": "Other / Profanity"},
    {"lang_group": "English", "id": "EN2-02", "text": "suck my cock and go fuck off", "expected_is_bully": True, "expected_category": "Other / Harassment"},
    {"lang_group": "English", "id": "EN2-03", "text": "your mother is a dirty whore", "expected_is_bully": True, "expected_category": "Gender / Other"},
    {"lang_group": "English", "id": "EN2-04", "text": "just kill yourself already kys loser", "expected_is_bully": True, "expected_category": "Other / Suicide Encouragement"},
    {"lang_group": "English", "id": "EN2-05", "text": "you are really helpful, thank you so much!", "expected_is_bully": False, "expected_category": "N/A (Safe)"},
    {"lang_group": "English", "id": "EN2-06", "text": "Shut up you wrinkled old senile hag boomer", "expected_is_bully": True, "expected_category": "Age"},
    {"lang_group": "English", "id": "EN2-07", "text": "All Muslims should be deported, they are terrorists", "expected_is_bully": True, "expected_category": "Religion"},
    {"lang_group": "English", "id": "EN2-08", "text": "Black people are all lazy and dumb", "expected_is_bully": True, "expected_category": "Ethnicity"},
    {"lang_group": "English", "id": "EN2-09", "text": "Women belong only in the kitchen not offices", "expected_is_bully": True, "expected_category": "Gender"},
    {"lang_group": "English", "id": "EN2-10", "text": "Old people are a waste and should die", "expected_is_bully": True, "expected_category": "Age"},
    {"lang_group": "English", "id": "EN2-11", "text": "You're a fat disgusting ugly bitch", "expected_is_bully": True, "expected_category": "Gender / Other"},
    {"lang_group": "English", "id": "EN2-12", "text": "Kill every Jew, they run the world", "expected_is_bully": True, "expected_category": "Religion"},
    {"lang_group": "English", "id": "EN2-13", "text": "Transgenders are sick freaks", "expected_is_bully": True, "expected_category": "Gender / Other"},
    {"lang_group": "English", "id": "EN2-14", "text": "Asians have tiny dicks and only good at studying", "expected_is_bully": True, "expected_category": "Ethnicity"},
    {"lang_group": "English", "id": "EN2-15", "text": "Christians are all brainwashed idiots", "expected_is_bully": True, "expected_category": "Religion"},
    {"lang_group": "English", "id": "EN2-16", "text": "You look like a little kid, go back to school", "expected_is_bully": True, "expected_category": "Age"},
    {"lang_group": "English", "id": "EN2-17", "text": "I hope you get raped and suffer", "expected_is_bully": True, "expected_category": "Gender / Severe Other"},
    {"lang_group": "English", "id": "EN2-18", "text": "Your whole family are worthless animals", "expected_is_bully": True, "expected_category": "Other"},
    {"lang_group": "English", "id": "EN2-19", "text": "Great work on the project, really impressive", "expected_is_bully": False, "expected_category": "N/A (Safe)"},
    {"lang_group": "English", "id": "EN2-20", "text": "Can you help me with this please? Thanks a lot", "expected_is_bully": False, "expected_category": "N/A (Safe)"},
    {"lang_group": "English", "id": "EN2-21", "text": "Have a wonderful day and stay safe", "expected_is_bully": False, "expected_category": "N/A (Safe)"},
    {"lang_group": "English", "id": "EN2-22", "text": "The weather looks nice today", "expected_is_bully": False, "expected_category": "N/A (Safe)"},
    {"lang_group": "English", "id": "EN2-23", "text": "I really like this product, recommend it", "expected_is_bully": False, "expected_category": "N/A (Safe)"},
    {"lang_group": "English", "id": "EN2-24", "text": "You're a total genius, keep going", "expected_is_bully": False, "expected_category": "N/A (Safe)"},
    {"lang_group": "English", "id": "EN2-25", "text": "go to hell you useless piece of shit", "expected_is_bully": True, "expected_category": "Other"},
    {"lang_group": "English", "id": "EN2-26", "text": "your father should have pulled out", "expected_is_bully": True, "expected_category": "Other / Harassment"},
    {"lang_group": "English", "id": "EN2-27", "text": "women are only good for sex", "expected_is_bully": True, "expected_category": "Gender"},
    {"lang_group": "English", "id": "EN2-28", "text": "old hags like you need to retire and shut up", "expected_is_bully": True, "expected_category": "Age"},
    {"lang_group": "English", "id": "EN2-29", "text": "all Indians are just scam callers", "expected_is_bully": True, "expected_category": "Ethnicity"},
    {"lang_group": "English", "id": "EN2-30", "text": "gay people are an abomination to society", "expected_is_bully": True, "expected_category": "Gender / Other"},
    {"lang_group": "English", "id": "EN2-31", "text": "I hope your whole bloodline gets destroyed", "expected_is_bully": True, "expected_category": "Other"},
    {"lang_group": "English", "id": "EN2-32", "text": "you deserve to be bullied every single day", "expected_is_bully": True, "expected_category": "Other"},
    {"lang_group": "English", "id": "EN2-33", "text": "Thanks for the useful feedback", "expected_is_bully": False, "expected_category": "N/A (Safe)"},
    {"lang_group": "English", "id": "EN2-34", "text": "This is just a normal work discussion", "expected_is_bully": False, "expected_category": "N/A (Safe)"},
    {"lang_group": "English", "id": "EN2-35", "text": "Looking forward to working with you", "expected_is_bully": False, "expected_category": "N/A (Safe)"},

    # Hinglish (35)
    {"lang_group": "Hinglish", "id": "HI-EN2-01", "text": "madarchod saale", "expected_is_bully": True, "expected_category": "Other / Profanity"},
    {"lang_group": "Hinglish", "id": "HI-EN2-02", "text": "bhosdiwale traffic wale", "expected_is_bully": True, "expected_category": "Other / Profanity"},
    {"lang_group": "Hinglish", "id": "HI-EN2-03", "text": "isko chappal nikaal ke maarna chahiye", "expected_is_bully": True, "expected_category": "Other / Harassment"},
    {"lang_group": "Hinglish", "id": "HI-EN2-04", "text": "bahut smart ho aap, dimaag use mat karna", "expected_is_bully": True, "expected_category": "Other / Sarcasm / Mockery"},
    {"lang_group": "Hinglish", "id": "HI-EN2-05", "text": "teri maa ki choot", "expected_is_bully": True, "expected_category": "Other / Gender / Profanity"},
    {"lang_group": "Hinglish", "id": "HI-EN2-06", "text": "saale madarchod mar jaa", "expected_is_bully": True, "expected_category": "Other / Harm / Profanity"},
    {"lang_group": "Hinglish", "id": "HI-EN2-07", "text": "behen ke lode tu kya soch raha", "expected_is_bully": True, "expected_category": "Other / Profanity"},
    {"lang_group": "Hinglish", "id": "HI-EN2-08", "text": "tu pure number ka harami hai", "expected_is_bully": True, "expected_category": "Other / Insult"},
    {"lang_group": "Hinglish", "id": "HI-EN2-09", "text": "randi ki aulaad", "expected_is_bully": True, "expected_category": "Gender / Other / Profanity"},
    {"lang_group": "Hinglish", "id": "HI-EN2-10", "text": "chutiya kahi ka", "expected_is_bully": True, "expected_category": "Other / Profanity"},
    {"lang_group": "Hinglish", "id": "HI-EN2-11", "text": "teri behen ko main chodunga", "expected_is_bully": True, "expected_category": "Gender / Severe Harassment"},
    {"lang_group": "Hinglish", "id": "HI-EN2-12", "text": "tu pure khandaan ka kalank hai", "expected_is_bully": True, "expected_category": "Other / Insult"},
    {"lang_group": "Hinglish", "id": "HI-EN2-13", "text": "buddhe log ghar pe hi baitho", "expected_is_bully": True, "expected_category": "Age"},
    {"lang_group": "Hinglish", "id": "HI-EN2-14", "text": "musalman log sab terrorist hain", "expected_is_bully": True, "expected_category": "Religion"},
    {"lang_group": "Hinglish", "id": "HI-EN2-15", "text": "hindu log pagal hote hain", "expected_is_bully": True, "expected_category": "Religion"},
    {"lang_group": "Hinglish", "id": "HI-EN2-16", "text": "ladkiyan sirf kitchen ke liye hi hain", "expected_is_bully": True, "expected_category": "Gender"},
    {"lang_group": "Hinglish", "id": "HI-EN2-17", "text": "black log bilkul lazy hote hain", "expected_is_bully": True, "expected_category": "Ethnicity"},
    {"lang_group": "Hinglish", "id": "HI-EN2-18", "text": "tu itna mota hai dekhne layak nahi", "expected_is_bully": True, "expected_category": "Other / Body Shaming"},
    {"lang_group": "Hinglish", "id": "HI-EN2-19", "text": "jaake mar jaa koi nahi roega", "expected_is_bully": True, "expected_category": "Other / Self-harm incitement"},
    {"lang_group": "Hinglish", "id": "HI-EN2-20", "text": "aap bahut acche ho thank you", "expected_is_bully": False, "expected_category": "N/A (Safe)"},
    {"lang_group": "Hinglish", "id": "HI-EN2-21", "text": "ye project aapne bahut accha banaya", "expected_is_bully": False, "expected_category": "N/A (Safe)"},
    {"lang_group": "Hinglish", "id": "HI-EN2-22", "text": "please meri help kar do thanks", "expected_is_bully": False, "expected_category": "N/A (Safe)"},
    {"lang_group": "Hinglish", "id": "HI-EN2-23", "text": "aaj ka mausam accha hai", "expected_is_bully": False, "expected_category": "N/A (Safe)"},
    {"lang_group": "Hinglish", "id": "HI-EN2-24", "text": "main aapki help kar sakta hoon", "expected_is_bully": False, "expected_category": "N/A (Safe)"},
    {"lang_group": "Hinglish", "id": "HI-EN2-25", "text": "isko toh chappal khol ke maar dena chahiye", "expected_is_bully": True, "expected_category": "Other / Harassment"},
    {"lang_group": "Hinglish", "id": "HI-EN2-26", "text": "isko to chappal nikaal ke marna chahiye", "expected_is_bully": True, "expected_category": "Other / Harassment"},
    {"lang_group": "Hinglish", "id": "HI-EN2-27", "text": "meri chappal kahan rakhi hai", "expected_is_bully": False, "expected_category": "N/A (Safe)"},
    {"lang_group": "Hinglish", "id": "HI-EN2-28", "text": "yeh chappal bohot comfortable lag rahi", "expected_is_bully": False, "expected_category": "N/A (Safe)"},
    {"lang_group": "Hinglish", "id": "HI-EN2-29", "text": "waah kya dimaag hai bhai, school gaye the kabhi?", "expected_is_bully": True, "expected_category": "Other / Sarcasm / Mockery"},
    {"lang_group": "Hinglish", "id": "HI-EN2-30", "text": "kya scene hai aaj toh time pe aa gaye", "expected_is_bully": False, "expected_category": "N/A (Safe/Mild Sarcasm)"},
    {"lang_group": "Hinglish", "id": "HI-EN2-31", "text": "tu pagal hai kya sala", "expected_is_bully": True, "expected_category": "Other / Insult (or Borderline)"},
    {"lang_group": "Hinglish", "id": "HI-EN2-32", "text": "saale gandu kahi ke", "expected_is_bully": True, "expected_category": "Other / Profanity"},
    {"lang_group": "Hinglish", "id": "HI-EN2-33", "text": "teri maa randi hai pure", "expected_is_bully": True, "expected_category": "Gender / Other / Profanity"},
    {"lang_group": "Hinglish", "id": "HI-EN2-34", "text": "aap bohot talented ho yaar", "expected_is_bully": False, "expected_category": "N/A (Safe)"},
    {"lang_group": "Hinglish", "id": "HI-EN2-35", "text": "is baat mein koi problem nahi hai", "expected_is_bully": False, "expected_category": "N/A (Safe)"},

    # Pure Hindi (30)
    {"lang_group": "Pure Hindi", "id": "HI2-01", "text": "मादरचोद साले", "expected_is_bully": True, "expected_category": "Other / Profanity"},
    {"lang_group": "Pure Hindi", "id": "HI2-02", "text": "भोसड़ीवाले ट्रैफिक वाले", "expected_is_bully": True, "expected_category": "Other / Profanity"},
    {"lang_group": "Pure Hindi", "id": "HI2-03", "text": "इसकी चप्पल निकाल के मारना चाहिए", "expected_is_bully": True, "expected_category": "Other / Harassment"},
    {"lang_group": "Pure Hindi", "id": "HI2-04", "text": "बहुत स्मार्ट हो आप, दिमाग यूज़ मत करना", "expected_is_bully": True, "expected_category": "Other / Sarcasm / Mockery"},
    {"lang_group": "Pure Hindi", "id": "HI2-05", "text": "तेरी माँ की चूत", "expected_is_bully": True, "expected_category": "Other / Gender / Profanity"},
    {"lang_group": "Pure Hindi", "id": "HI2-06", "text": "साले मादरचोद मर जा", "expected_is_bully": True, "expected_category": "Other / Harm / Profanity"},
    {"lang_group": "Pure Hindi", "id": "HI2-07", "text": "बहन के लोड़े तू क्या सोच रहा", "expected_is_bully": True, "expected_category": "Other / Profanity"},
    {"lang_group": "Pure Hindi", "id": "HI2-08", "text": "तू पूरे नंबर का हरामी है", "expected_is_bully": True, "expected_category": "Other / Insult"},
    {"lang_group": "Pure Hindi", "id": "HI2-09", "text": "रंडी की औलाद", "expected_is_bully": True, "expected_category": "Gender / Other / Profanity"},
    {"lang_group": "Pure Hindi", "id": "HI2-10", "text": "चुतिया कहीं का", "expected_is_bully": True, "expected_category": "Other / Profanity"},
    {"lang_group": "Pure Hindi", "id": "HI2-11", "text": "तेरी बहन को मैं चोदूंगा", "expected_is_bully": True, "expected_category": "Gender / Severe Harassment"},
    {"lang_group": "Pure Hindi", "id": "HI2-12", "text": "तू पूरे खानदान का कलंक है", "expected_is_bully": True, "expected_category": "Other / Insult"},
    {"lang_group": "Pure Hindi", "id": "HI2-13", "text": "बूढ़े लोग घर पे ही बैठो", "expected_is_bully": True, "expected_category": "Age"},
    {"lang_group": "Pure Hindi", "id": "HI2-14", "text": "मुसलमान लोग सब आतंकवादी हैं", "expected_is_bully": True, "expected_category": "Religion"},
    {"lang_group": "Pure Hindi", "id": "HI2-15", "text": "हिंदू लोग पागल होते हैं", "expected_is_bully": True, "expected_category": "Religion"},
    {"lang_group": "Pure Hindi", "id": "HI2-16", "text": "लड़कियां सिर्फ किचन के लिए ही हैं", "expected_is_bully": True, "expected_category": "Gender"},
    {"lang_group": "Pure Hindi", "id": "HI2-17", "text": "काले लोग बिल्कुल आलसी होते हैं", "expected_is_bully": True, "expected_category": "Ethnicity"},
    {"lang_group": "Pure Hindi", "id": "HI2-18", "text": "तू इतना मोटा है देखने लायक नहीं", "expected_is_bully": True, "expected_category": "Other / Body Shaming"},
    {"lang_group": "Pure Hindi", "id": "HI2-19", "text": "जाके मर जा कोई नहीं रोएगा", "expected_is_bully": True, "expected_category": "Other / Self-harm incitement"},
    {"lang_group": "Pure Hindi", "id": "HI2-20", "text": "आप बहुत अच्छे हो धन्यवाद", "expected_is_bully": False, "expected_category": "N/A (Safe)"},
    {"lang_group": "Pure Hindi", "id": "HI2-21", "text": "यह प्रोजेक्ट आपने बहुत अच्छा बनाया", "expected_is_bully": False, "expected_category": "N/A (Safe)"},
    {"lang_group": "Pure Hindi", "id": "HI2-22", "text": "कृपया मेरी मदद कर दो धन्यवाद", "expected_is_bully": False, "expected_category": "N/A (Safe)"},
    {"lang_group": "Pure Hindi", "id": "HI2-23", "text": "आज का मौसम अच्छा है", "expected_is_bully": False, "expected_category": "N/A (Safe)"},
    {"lang_group": "Pure Hindi", "id": "HI2-24", "text": "मैं आपकी मदद कर सकता हूँ", "expected_is_bully": False, "expected_category": "N/A (Safe)"},
    {"lang_group": "Pure Hindi", "id": "HI2-25", "text": "इसकी तो चप्पल खोल के मार देना चाहिए", "expected_is_bully": True, "expected_category": "Other / Harassment"},
    {"lang_group": "Pure Hindi", "id": "HI2-26", "text": "मेरी चप्पल कहाँ रखी है", "expected_is_bully": False, "expected_category": "N/A (Safe)"},
    {"lang_group": "Pure Hindi", "id": "HI2-27", "text": "यह चप्पल बहुत आरामदायक लग रही", "expected_is_bully": False, "expected_category": "N/A (Safe)"},
    {"lang_group": "Pure Hindi", "id": "HI2-28", "text": "वाह क्या दिमाग है भाई, स्कूल गए थे कभी?", "expected_is_bully": True, "expected_category": "Other / Sarcasm / Mockery"},
    {"lang_group": "Pure Hindi", "id": "HI2-29", "text": "क्या सीन है आज तो टाइम पे आ गए", "expected_is_bully": False, "expected_category": "N/A (Safe/Mild Sarcasm)"},
    {"lang_group": "Pure Hindi", "id": "HI2-30", "text": "तू पागल है क्या साला", "expected_is_bully": True, "expected_category": "Other / Insult (or Borderline)"}
]

def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    print("Loading classifier...")
    classifier = get_classifier()
    print("Running inference across 100 new similar test cases...")

    results = []
    t_start = time.time()

    for i, item in enumerate(new_test_cases_data, 1):
        text = item["text"]
        muril_res = classifier.predict(text, model_choice="muril")
        baseline_res = classifier.predict(text, model_choice="baseline")

        m_bully = muril_res.get("is_cyberbullying")
        e_bully = item["expected_is_bully"]
        if m_bully == e_bully:
            eval_status = "MATCH"
        elif m_bully and not e_bully:
            eval_status = "FALSE_POSITIVE"
        else:
            eval_status = "FALSE_NEGATIVE"

        res_entry = {
            "index": i,
            "id": item["id"],
            "language_group": item["lang_group"],
            "text": text,
            "expected_is_bully": item["expected_is_bully"],
            "expected_category": item["expected_category"],
            "eval_status": eval_status,
            "muril": {
                "verdict": muril_res.get("verdict"),
                "is_cyberbullying": muril_res.get("is_cyberbullying"),
                "category": muril_res.get("category"),
                "confidence": muril_res.get("confidence"),
                "model_used": muril_res.get("model_used"),
                "all_probabilities": muril_res.get("all_probabilities"),
                "safety_net_triggered": muril_res.get("safety_net_note") is not None or "Safety-Net" in muril_res.get("model_used", ""),
                "latency_ms": muril_res.get("latency_ms"),
                "detected_lang": muril_res.get("language"),
                "trigger_words": muril_res.get("explainability", {}).get("trigger_words", [])
            },
            "baseline": {
                "verdict": baseline_res.get("verdict"),
                "is_cyberbullying": baseline_res.get("is_cyberbullying"),
                "category": baseline_res.get("category"),
                "confidence": baseline_res.get("confidence"),
                "all_probabilities": baseline_res.get("all_probabilities")
            }
        }
        results.append(res_entry)
        status_symbol = "MATCH" if eval_status == "MATCH" else f"DIFF ({eval_status})"
        print(f"[{i:03d}/100] [{item['lang_group']}] [{status_symbol}] Verdict: {muril_res.get('verdict')} | Conf: {muril_res.get('confidence')}% | ExpBully: {e_bully}")

    elapsed = round(time.time() - t_start, 2)
    print(f"\nCompleted in {elapsed}s.")

    out_file = os.path.join(ROOT_DIR, "new_test_cases_evaluation.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved new raw results to {out_file}")

if __name__ == "__main__":
    main()
