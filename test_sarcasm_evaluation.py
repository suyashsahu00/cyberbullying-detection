"""
Empirical test suite for evaluating model behavior on implicit sarcastic Hinglish comments.
Tests both sarcastic insults (implicit cyberbullying) and benign sarcasm / teasing.
"""
from src.model import CyberbullyingSystem

def run_sarcasm_eval():
    print("Loading CyberbullyingSystem (MuRIL v2)...")
    system = CyberbullyingSystem()

    # 12 diverse sarcastic Hinglish samples:
    # 6 implicit abusive/bullying sarcasm (mockery, passive-aggressive insult, degradation)
    # 6 benign/witty sarcasm or friendly teasing (not cyberbullying)
    test_cases = [
        # Implicit Bullying / Harassment (True: Cyberbullying)
        {
            "text": "bohot samajhdar ho aap, dimaag mat use karna",
            "true_label": "cyberbullying",
            "type": "Mock intelligence / Insult"
        },
        {
            "text": "waah kya logic hai bhai, school kabhi gaye bhi the kya?",
            "true_label": "cyberbullying",
            "type": "Belittling education / Mockery"
        },
        {
            "text": "itna sundar chehra hai aapka ki mirror bhi sharma jaye",
            "true_label": "cyberbullying",
            "type": "Appearance sarcasm"
        },
        {
            "text": "aapse bada gyaani toh puri duniya mein koi paida hi nahi hua",
            "true_label": "cyberbullying",
            "type": "Mocking arrogance / Belittling"
        },
        {
            "text": "aapki acting dekh ke oscar committee ne rona shuru kar diya",
            "true_label": "cyberbullying",
            "type": "Derisive critique / Public humiliation"
        },
        {
            "text": "kripya chup rahiye, aapke bolne se desh ka IQ kam ho raha hai",
            "true_label": "cyberbullying",
            "type": "Hostile shaming / Demeaning"
        },
        # Benign Sarcasm / Friendly teasing / Banter (True: Safe / not_cyberbullying)
        {
            "text": "haan bhai tum toh ambani ke sage bhai ho na jo itna ud rahe ho",
            "true_label": "not_cyberbullying",
            "type": "Friendly banter / Meme joke"
        },
        {
            "text": "kya baat hai aaj toh time pe aa gaye, suraj kidhar se nikla tha?",
            "true_label": "not_cyberbullying",
            "type": "Common sarcastic compliment / Teasing"
        },
        {
            "text": "waah bhai itni mehnat karke exam mein 2 number leke aaye",
            "true_label": "not_cyberbullying",
            "type": "Self/peer mild irony"
        },
        {
            "text": "kitna kaam karte ho yaar, thoda aaram bhi kar liya karo lazy insaan",
            "true_label": "not_cyberbullying",
            "type": "Affectionate teasing"
        },
        {
            "text": "sahi khel gaye guru, poora din soye aur fir bhi thak gaye",
            "true_label": "not_cyberbullying",
            "type": "Humorous irony"
        },
        {
            "text": "arre scientist saab, thoda hum anpadhon ko bhi gyaan de do",
            "true_label": "not_cyberbullying",
            "type": "Playful self-deprecation"
        }
    ]

    print("\n" + "="*80)
    print("EMPIRICAL TEST: IMPLICIT SARCASM EVALUATION ON GOOGLE MuRIL v2")
    print("="*80)

    correct_muril = 0
    results = []

    for idx, item in enumerate(test_cases, 1):
        pred_muril = system.predict_muril(item["text"])
        is_muril_cb = pred_muril["is_cyberbullying"]
        is_muril_correct = (is_muril_cb and item["true_label"] == "cyberbullying") or \
                           (not is_muril_cb and item["true_label"] == "not_cyberbullying")
        if is_muril_correct:
            correct_muril += 1

        results.append({
            "idx": idx,
            "text": item["text"],
            "type": item["type"],
            "true_label": item["true_label"],
            "category": pred_muril["category"],
            "verdict": pred_muril["verdict"],
            "is_cb": is_muril_cb,
            "confidence": pred_muril["confidence"],
            "correct": is_muril_correct
        })

        status = "[PASS]" if is_muril_correct else "[FAIL]"
        print(f"[{idx:02d}] {status:6s} | True: {item['true_label']:16s} | Verdict: {pred_muril['verdict']:24s} ({pred_muril['confidence']:.1f}%) | Category: {pred_muril['category']:10s} | Text: {item['text']}")

    total = len(test_cases)
    accuracy = (correct_muril / total) * 100
    print("\n" + "="*80)
    print(f"SUMMARY: Sarcasm Mini-Suite Accuracy: {correct_muril}/{total} ({accuracy:.1f}%)")
    print("="*80)

    # Save to JSON
    import json
    with open("models/sarcasm_eval_results.json", "w", encoding="utf-8") as f:
        json.dump({"total": total, "correct": correct_muril, "accuracy": accuracy, "details": results}, f, indent=2)

if __name__ == "__main__":
    run_sarcasm_eval()
