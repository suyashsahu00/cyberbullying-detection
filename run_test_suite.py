import os
import sys
import json
import time

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.model import get_classifier

test_cases_data = [
    # English (35)
    {"lang_group": "English", "id": "EN-01", "text": "you fucking idiot / motherfucker", "expected_is_bully": True, "expected_category": "Other / Profanity"},
    {"lang_group": "English", "id": "EN-02", "text": "suck my dick, go fuck yourself", "expected_is_bully": True, "expected_category": "Other / Harassment"},
    {"lang_group": "English", "id": "EN-03", "text": "your mom is a whore", "expected_is_bully": True, "expected_category": "Gender / Other"},
    {"lang_group": "English", "id": "EN-04", "text": "go kill yourself loser kys", "expected_is_bully": True, "expected_category": "Other / Suicide Encouragement"},
    {"lang_group": "English", "id": "EN-05", "text": "you are so helpful, thank you!", "expected_is_bully": False, "expected_category": "N/A (Safe)"},
    {"lang_group": "English", "id": "EN-06", "text": "Shut up you wrinkly senile old hag boomer!", "expected_is_bully": True, "expected_category": "Age"},
    {"lang_group": "English", "id": "EN-07", "text": "All Muslims are terrorists and should be banned", "expected_is_bully": True, "expected_category": "Religion"},
    {"lang_group": "English", "id": "EN-08", "text": "Black people are lazy and stupid", "expected_is_bully": True, "expected_category": "Ethnicity"},
    {"lang_group": "English", "id": "EN-09", "text": "Girls belong in the kitchen, not in tech", "expected_is_bully": True, "expected_category": "Gender"},
    {"lang_group": "English", "id": "EN-10", "text": "Old people should just die already, waste of space", "expected_is_bully": True, "expected_category": "Age"},
    {"lang_group": "English", "id": "EN-11", "text": "You're a fat ugly bitch nobody wants", "expected_is_bully": True, "expected_category": "Gender / Other"},
    {"lang_group": "English", "id": "EN-12", "text": "Kill all the Jews, they control everything", "expected_is_bully": True, "expected_category": "Religion / Ethnicity"},
    {"lang_group": "English", "id": "EN-13", "text": "Trans people are mentally ill freaks", "expected_is_bully": True, "expected_category": "Gender / Other"},
    {"lang_group": "English", "id": "EN-14", "text": "Asian people have small dicks and are good at math only", "expected_is_bully": True, "expected_category": "Ethnicity"},
    {"lang_group": "English", "id": "EN-15", "text": "Christians are brainwashed sheep", "expected_is_bully": True, "expected_category": "Religion"},
    {"lang_group": "English", "id": "EN-16", "text": "You look like a 12 year old, go back to school kid", "expected_is_bully": True, "expected_category": "Age"},
    {"lang_group": "English", "id": "EN-17", "text": "I hope you get raped and die alone", "expected_is_bully": True, "expected_category": "Gender / Severe Other"},
    {"lang_group": "English", "id": "EN-18", "text": "Your whole family is a bunch of worthless animals", "expected_is_bully": True, "expected_category": "Other"},
    {"lang_group": "English", "id": "EN-19", "text": "Nice job on the presentation, really well done", "expected_is_bully": False, "expected_category": "N/A (Safe)"},
    {"lang_group": "English", "id": "EN-20", "text": "Can you please help me with this code? Thanks!", "expected_is_bully": False, "expected_category": "N/A (Safe)"},
    {"lang_group": "English", "id": "EN-21", "text": "Have a great day, stay safe", "expected_is_bully": False, "expected_category": "N/A (Safe)"},
    {"lang_group": "English", "id": "EN-22", "text": "The weather is nice today, isn't it?", "expected_is_bully": False, "expected_category": "N/A (Safe)"},
    {"lang_group": "English", "id": "EN-23", "text": "I love this product, highly recommend it", "expected_is_bully": False, "expected_category": "N/A (Safe)"},
    {"lang_group": "English", "id": "EN-24", "text": "You're an absolute genius, keep up the good work", "expected_is_bully": False, "expected_category": "N/A (Safe)"},
    {"lang_group": "English", "id": "EN-25", "text": "go to hell you piece of shit", "expected_is_bully": True, "expected_category": "Other"},
    {"lang_group": "English", "id": "EN-26", "text": "your dad should have pulled out", "expected_is_bully": True, "expected_category": "Other / Harassment"},
    {"lang_group": "English", "id": "EN-27", "text": "women are only good for one thing", "expected_is_bully": True, "expected_category": "Gender"},
    {"lang_group": "English", "id": "EN-28", "text": "old hags like you should retire and shut up", "expected_is_bully": True, "expected_category": "Age"},
    {"lang_group": "English", "id": "EN-29", "text": "all Indians are scam artists", "expected_is_bully": True, "expected_category": "Ethnicity"},
    {"lang_group": "English", "id": "EN-30", "text": "queer people are an abomination", "expected_is_bully": True, "expected_category": "Gender / Religion / Other"},
    {"lang_group": "English", "id": "EN-31", "text": "I hope your entire bloodline suffers", "expected_is_bully": True, "expected_category": "Other"},
    {"lang_group": "English", "id": "EN-32", "text": "you deserve to be bullied every day", "expected_is_bully": True, "expected_category": "Other"},
    {"lang_group": "English", "id": "EN-33", "text": "Thanks for the feedback, I appreciate it", "expected_is_bully": False, "expected_category": "N/A (Safe)"},
    {"lang_group": "English", "id": "EN-34", "text": "This is a normal professional discussion", "expected_is_bully": False, "expected_category": "N/A (Safe)"},
    {"lang_group": "English", "id": "EN-35", "text": "Looking forward to collaborating with you", "expected_is_bully": False, "expected_category": "N/A (Safe)"},

    # Hinglish (35)
    {"lang_group": "Hinglish", "id": "HI-EN-01", "text": "madarchod", "expected_is_bully": True, "expected_category": "Other / Profanity"},
    {"lang_group": "Hinglish", "id": "HI-EN-02", "text": "bhosdiwala traffic", "expected_is_bully": True, "expected_category": "Other / Profanity (Non-targeted/Targeted)"},
    {"lang_group": "Hinglish", "id": "HI-EN-03", "text": "isko to chappal khol k maarna chaiye", "expected_is_bully": True, "expected_category": "Other / Harassment"},
    {"lang_group": "Hinglish", "id": "HI-EN-04", "text": "bohot samajhdar ho aap, dimaag mat use karna", "expected_is_bully": True, "expected_category": "Other / Sarcasm / Mockery"},
    {"lang_group": "Hinglish", "id": "HI-EN-05", "text": "teri maa ki chut", "expected_is_bully": True, "expected_category": "Other / Gender / Profanity"},
    {"lang_group": "Hinglish", "id": "HI-EN-06", "text": "saale madarchod jaa mar", "expected_is_bully": True, "expected_category": "Other / Harm / Profanity"},
    {"lang_group": "Hinglish", "id": "HI-EN-07", "text": "behenchod tu kya sochta hai", "expected_is_bully": True, "expected_category": "Other / Profanity"},
    {"lang_group": "Hinglish", "id": "HI-EN-08", "text": "tu ek number ka harami hai", "expected_is_bully": True, "expected_category": "Other / Insult"},
    {"lang_group": "Hinglish", "id": "HI-EN-09", "text": "randi ki aulad", "expected_is_bully": True, "expected_category": "Gender / Other / Profanity"},
    {"lang_group": "Hinglish", "id": "HI-EN-10", "text": "chutiya sala", "expected_is_bully": True, "expected_category": "Other / Profanity"},
    {"lang_group": "Hinglish", "id": "HI-EN-11", "text": "teri behen ko chodunga", "expected_is_bully": True, "expected_category": "Gender / Severe Harassment"},
    {"lang_group": "Hinglish", "id": "HI-EN-12", "text": "tu to pure family ka kalank hai", "expected_is_bully": True, "expected_category": "Other / Insult"},
    {"lang_group": "Hinglish", "id": "HI-EN-13", "text": "buddhe logon ko ghar pe baithna chahiye", "expected_is_bully": True, "expected_category": "Age"},
    {"lang_group": "Hinglish", "id": "HI-EN-14", "text": "musalman sab terrorist hote hain", "expected_is_bully": True, "expected_category": "Religion"},
    {"lang_group": "Hinglish", "id": "HI-EN-15", "text": "hindu log paagal hain", "expected_is_bully": True, "expected_category": "Religion"},
    {"lang_group": "Hinglish", "id": "HI-EN-16", "text": "ladkiyan sirf kitchen ke liye bani hain", "expected_is_bully": True, "expected_category": "Gender"},
    {"lang_group": "Hinglish", "id": "HI-EN-17", "text": "black log lazy hote hain", "expected_is_bully": True, "expected_category": "Ethnicity"},
    {"lang_group": "Hinglish", "id": "HI-EN-18", "text": "tu itna mota hai, dekhne layak nahi", "expected_is_bully": True, "expected_category": "Other / Body Shaming"},
    {"lang_group": "Hinglish", "id": "HI-EN-19", "text": "jaa mar ja, koi nahi roega", "expected_is_bully": True, "expected_category": "Other / Self-harm incitement"},
    {"lang_group": "Hinglish", "id": "HI-EN-20", "text": "aap bohot acche ho, thanks", "expected_is_bully": False, "expected_category": "N/A (Safe)"},
    {"lang_group": "Hinglish", "id": "HI-EN-21", "text": "yeh project bohot accha banaya hai aapne", "expected_is_bully": False, "expected_category": "N/A (Safe)"},
    {"lang_group": "Hinglish", "id": "HI-EN-22", "text": "please help me with this, thank you", "expected_is_bully": False, "expected_category": "N/A (Safe)"},
    {"lang_group": "Hinglish", "id": "HI-EN-23", "text": "aaj mausam accha hai", "expected_is_bully": False, "expected_category": "N/A (Safe)"},
    {"lang_group": "Hinglish", "id": "HI-EN-24", "text": "main aapki madad kar sakta hoon", "expected_is_bully": False, "expected_category": "N/A (Safe)"},
    {"lang_group": "Hinglish", "id": "HI-EN-25", "text": "isko toh chappal khol ke marna chahiye", "expected_is_bully": True, "expected_category": "Other / Harassment"},
    {"lang_group": "Hinglish", "id": "HI-EN-26", "text": "isko to chappal khol ke marna chahiye", "expected_is_bully": True, "expected_category": "Other / Harassment"},
    {"lang_group": "Hinglish", "id": "HI-EN-27", "text": "meri chappal kahan hai", "expected_is_bully": False, "expected_category": "N/A (Safe)"},
    {"lang_group": "Hinglish", "id": "HI-EN-28", "text": "yeh chappal bahut comfortable hai", "expected_is_bully": False, "expected_category": "N/A (Safe)"},
    {"lang_group": "Hinglish", "id": "HI-EN-29", "text": "waah kya logic hai bhai, school kabhi gaye bhi the kya?", "expected_is_bully": True, "expected_category": "Other / Sarcasm / Mockery"},
    {"lang_group": "Hinglish", "id": "HI-EN-30", "text": "kya baat hai aaj toh time pe aa gaye", "expected_is_bully": False, "expected_category": "N/A (Safe/Mild Sarcasm)"},
    {"lang_group": "Hinglish", "id": "HI-EN-31", "text": "tu pagal hai kya", "expected_is_bully": True, "expected_category": "Other / Insult (or Borderline)"},
    {"lang_group": "Hinglish", "id": "HI-EN-32", "text": "saale gandu", "expected_is_bully": True, "expected_category": "Other / Profanity"},
    {"lang_group": "Hinglish", "id": "HI-EN-33", "text": "teri maa randi hai", "expected_is_bully": True, "expected_category": "Gender / Other / Profanity"},
    {"lang_group": "Hinglish", "id": "HI-EN-34", "text": "aap bohot talented ho", "expected_is_bully": False, "expected_category": "N/A (Safe)"},
    {"lang_group": "Hinglish", "id": "HI-EN-35", "text": "is discussion mein koi problem nahi hai", "expected_is_bully": False, "expected_category": "N/A (Safe)"},

    # Pure Hindi (30)
    {"lang_group": "Pure Hindi", "id": "HI-01", "text": "मादरचोद", "expected_is_bully": True, "expected_category": "Other / Profanity"},
    {"lang_group": "Pure Hindi", "id": "HI-02", "text": "भोसड़ीवाले", "expected_is_bully": True, "expected_category": "Other / Profanity"},
    {"lang_group": "Pure Hindi", "id": "HI-03", "text": "इसकी तो चप्पल खोल के मारना चाहिए", "expected_is_bully": True, "expected_category": "Other / Harassment"},
    {"lang_group": "Pure Hindi", "id": "HI-04", "text": "बहुत समझदार हो आप, दिमाग मत यूज़ करना", "expected_is_bully": True, "expected_category": "Other / Sarcasm / Mockery"},
    {"lang_group": "Pure Hindi", "id": "HI-05", "text": "तेरी माँ की चुत", "expected_is_bully": True, "expected_category": "Other / Gender / Profanity"},
    {"lang_group": "Pure Hindi", "id": "HI-06", "text": "साले मादरचोद जा मर", "expected_is_bully": True, "expected_category": "Other / Harm / Profanity"},
    {"lang_group": "Pure Hindi", "id": "HI-07", "text": "बहनचोद तू क्या सोचता है", "expected_is_bully": True, "expected_category": "Other / Profanity"},
    {"lang_group": "Pure Hindi", "id": "HI-08", "text": "तू एक नंबर का हरामी है", "expected_is_bully": True, "expected_category": "Other / Insult"},
    {"lang_group": "Pure Hindi", "id": "HI-09", "text": "रंडी की औलाद", "expected_is_bully": True, "expected_category": "Gender / Other / Profanity"},
    {"lang_group": "Pure Hindi", "id": "HI-10", "text": "चुतिया साला", "expected_is_bully": True, "expected_category": "Other / Profanity"},
    {"lang_group": "Pure Hindi", "id": "HI-11", "text": "तेरी बहन को चोदूंगा", "expected_is_bully": True, "expected_category": "Gender / Severe Harassment"},
    {"lang_group": "Pure Hindi", "id": "HI-12", "text": "तू तो पूरे परिवार का कलंक है", "expected_is_bully": True, "expected_category": "Other / Insult"},
    {"lang_group": "Pure Hindi", "id": "HI-13", "text": "बूढ़े लोगों को घर पे बैठना चाहिए", "expected_is_bully": True, "expected_category": "Age"},
    {"lang_group": "Pure Hindi", "id": "HI-14", "text": "मुसलमान सब आतंकवादी होते हैं", "expected_is_bully": True, "expected_category": "Religion"},
    {"lang_group": "Pure Hindi", "id": "HI-15", "text": "हिंदू लोग पागल हैं", "expected_is_bully": True, "expected_category": "Religion"},
    {"lang_group": "Pure Hindi", "id": "HI-16", "text": "लड़कियां सिर्फ किचन के लिए बनी हैं", "expected_is_bully": True, "expected_category": "Gender"},
    {"lang_group": "Pure Hindi", "id": "HI-17", "text": "काले लोग आलसी होते हैं", "expected_is_bully": True, "expected_category": "Ethnicity"},
    {"lang_group": "Pure Hindi", "id": "HI-18", "text": "तू इतना मोटा है, देखने लायक नहीं", "expected_is_bully": True, "expected_category": "Other / Body Shaming"},
    {"lang_group": "Pure Hindi", "id": "HI-19", "text": "जा मर जा, कोई नहीं रोएगा", "expected_is_bully": True, "expected_category": "Other / Self-harm incitement"},
    {"lang_group": "Pure Hindi", "id": "HI-20", "text": "आप बहुत अच्छे हो, धन्यवाद", "expected_is_bully": False, "expected_category": "N/A (Safe)"},
    {"lang_group": "Pure Hindi", "id": "HI-21", "text": "यह प्रोजेक्ट बहुत अच्छा बनाया है आपने", "expected_is_bully": False, "expected_category": "N/A (Safe)"},
    {"lang_group": "Pure Hindi", "id": "HI-22", "text": "कृपया मेरी मदद करें, धन्यवाद", "expected_is_bully": False, "expected_category": "N/A (Safe)"},
    {"lang_group": "Pure Hindi", "id": "HI-23", "text": "आज मौसम अच्छा है", "expected_is_bully": False, "expected_category": "N/A (Safe)"},
    {"lang_group": "Pure Hindi", "id": "HI-24", "text": "मैं आपकी मदद कर सकता हूँ", "expected_is_bully": False, "expected_category": "N/A (Safe)"},
    {"lang_group": "Pure Hindi", "id": "HI-25", "text": "इसकी तो चप्पल खोल के मारना चाहिए", "expected_is_bully": True, "expected_category": "Other / Harassment"},
    {"lang_group": "Pure Hindi", "id": "HI-26", "text": "मेरी चप्पल कहाँ है", "expected_is_bully": False, "expected_category": "N/A (Safe)"},
    {"lang_group": "Pure Hindi", "id": "HI-27", "text": "यह चप्पल बहुत आरामदायक है", "expected_is_bully": False, "expected_category": "N/A (Safe)"},
    {"lang_group": "Pure Hindi", "id": "HI-28", "text": "वाह क्या लॉजिक है भाई, स्कूल कभी गए भी थे क्या?", "expected_is_bully": True, "expected_category": "Other / Sarcasm / Mockery"},
    {"lang_group": "Pure Hindi", "id": "HI-29", "text": "क्या बात है आज तो टाइम पे आ गए", "expected_is_bully": False, "expected_category": "N/A (Safe/Mild Sarcasm)"},
    {"lang_group": "Pure Hindi", "id": "HI-30", "text": "तू पागल है क्या", "expected_is_bully": True, "expected_category": "Other / Insult (or Borderline)"}
]

def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    print(f"Loading classifier...")
    classifier = get_classifier()
    print("Running inference across all 100 test cases...")
    
    results = []
    t_start = time.time()
    
    for i, item in enumerate(test_cases_data, 1):
        text = item["text"]
        
        # Primary MuRIL prediction
        muril_res = classifier.predict(text, model_choice="muril")
        
        # Baseline prediction for comparison
        baseline_res = classifier.predict(text, model_choice="baseline")
        
        # Alignment check
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
    
    out_file = os.path.join(ROOT_DIR, "test_cases_evaluation.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved raw results to {out_file}")

if __name__ == "__main__":
    main()
