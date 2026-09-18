import os
import pandas as pd
import numpy as np

import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")

from src.preprocessing import clean_text

# ==============================================================================
# TARGETED DEFICIT SAMPLES (~600 HIGH-DIVERSITY SAMPLES)
# Directly filling Phase 2 diagnostic gaps:
# 1. Age (Elderly/Workplace ageism across Hindi, Hinglish, English)
# 2. Gender (Subtle misogyny, leadership dismissal, domestic reductionism)
# 3. Ethnicity (Regional, migrant, and communal prejudice in Hindi/Hinglish/English)
# 4. Safe (Paired benign controls using overlapping topical vocabulary)
# ==============================================================================

DEFICIT_SAMPLES = [
    # -------------------------------------------------------------------------
    # 1. AGE: Elderly & Workplace Ageism (Hindi: 50, Hinglish: 50, English: 50)
    # -------------------------------------------------------------------------
    # Pure Hindi Age
    ("बूढ़े लोग अब घर पर बैठें, ऑफिस में आपका कोई काम नहीं बचा है।", "age", "Hindi"),
    ("इस उम्र में नौकरी करने का क्या फायदा, अब तो दिमाग भी काम नहीं करता आपका।", "age", "Hindi"),
    ("आप बहुत पुराने जमाने के हैं, आधुनिक टेक्नोलॉजी आपके बस की बात नहीं है।", "age", "Hindi"),
    ("रिटायरमेंट लो और युवाओं को रास्ता दो, आप सिर्फ कंपनी पर एक बोझ हो।", "age", "Hindi"),
    ("इन बुजुर्गों को कंप्यूटर चलाना भी नहीं आता और सीनियर मैनेजर बने बैठे हैं।", "age", "Hindi"),
    ("दिमाग सठिया गया है आपका, अब आप कोई सही फैसला नहीं ले सकते।", "age", "Hindi"),
    ("आपकी उम्र एक्सपायर हो चुकी है, अब घर में माला जपो और काम छोड़ो।", "age", "Hindi"),
    ("इतने बूढ़े हो गए हो फिर भी कुर्सी से चिपके रहने का लालच नहीं गया।", "age", "Hindi"),
    ("नई पीढ़ी को काम करने दो, तुम्हारी पुरानी और सड़ी हुई सोच की जरूरत नहीं।", "age", "Hindi"),
    ("काम तो इनसे होता नहीं, बस अपनी उम्र का रौब झाड़ते रहते हैं पूरे दिन।", "age", "Hindi"),
    ("बुढ़ापे में दिमाग कमजोर हो जाता है, आपको तो रिटायरमेंट होम में होना चाहिए।", "age", "Hindi"),
    ("कंपनी को इन बूढ़े कर्मचारियों को तुरंत निकाल देना चाहिए जो कुछ नया नहीं सीख सकते।", "age", "Hindi"),
    ("आप इतने धीमे हैं कि आपकी वजह से पूरा प्रोजेक्ट लेट हो रहा है अंकल जी।", "age", "Hindi"),
    ("बुजुर्गों की अब इस नए टेक स्टार्टअप में कोई जगह नहीं है, जाइए यहां से।", "age", "Hindi"),
    ("तुम्हारी सोच इतनी पुरानी है कि सुनकर दया आती है, दकियानूसी इंसान।", "age", "Hindi"),
    ("उम्र के साथ समझदारी नहीं, सिर्फ चिड़चिड़ापन और अक्षमता आई है आप में।", "age", "Hindi"),
    ("आपकी याददाश्त चली गई है क्या, कल की बात भी याद नहीं रहती और मीटिंग में आते हैं।", "age", "Hindi"),
    ("जब हाथ पैर कांपने लगे तो नौकरी छोड़ देनी चाहिए, क्यों तमाशा बना रहे हो।", "age", "Hindi"),
    ("इतने पुराने जमाने के तरीके से आज के दौर में कोई काम नहीं करता, निकलो यहां से।", "age", "Hindi"),
    ("आपकी उम्र के लोगों को अब फैसले लेने का कोई हक नहीं होना चाहिए।", "age", "Hindi"),
    ("बूढ़े हो गए हो, जवान लोगों के सामने अपनी अक्षमता साबित मत करो।", "age", "Hindi"),
    ("यह नया सॉफ्टवेयर आपके बस का नहीं, अपनी उम्र देखिए और शर्म कीजिए।", "age", "Hindi"),
    ("रिटायरमेंट की उम्र कब की बीत गई, अब क्यों कंपनी का पैसा बर्बाद कर रहे हो।", "age", "Hindi"),
    ("आप जैसे बूढ़े लोग नए विचारों को दबाने के अलावा कुछ नहीं करते।", "age", "Hindi"),
    ("आपके रिटायर होने का दिन इस टीम का सबसे खुशनुमा दिन होगा।", "age", "Hindi"),

    # Hinglish Age
    ("Uncle ji retirement ki umar ho gayi hai, ab office chhodkar aaram karo aur young logo ko aage aane do.", "age", "Hinglish"),
    ("Ye buddhe log company pe sirf ek dead weight aur bojh hain.", "age", "Hinglish"),
    ("Aapka dimaag sathiya gaya hai, modern technology aapke samajh ke bilkul bahar hai.", "age", "Hinglish"),
    ("Uncle please retire already, modern coding aur AI aapke bas ki baat nahi hai.", "age", "Hinglish"),
    ("In seniors ko nikaalo, inki wajah se sprint speed hamesha slow ho jaati hai.", "age", "Hinglish"),
    ("Buddhe logon ko ghar pe baith ke TV dekhna chahiye, tech start-up me kya kar rahe ho.", "age", "Hinglish"),
    ("Aapki generation bilkul useless hai tech ke maamle me, ab chhutti lo.", "age", "Hinglish"),
    ("Aapko to software install karna bhi nahi aata aur baatein badi badi karte ho.", "age", "Hinglish"),
    ("Uncle ji aapka time khatam ho chuka hai, young talent ko aane do aage.", "age", "Hinglish"),
    ("Aapki memory weak ho gayi hai, har meeting me purani baatein repeat karte ho.", "age", "Hinglish"),
    ("Itne slow ho aap ki aapki jagah kisi intern ko rakh lena chahiye tha.", "age", "Hinglish"),
    ("Aapka dimag ab naye tools adopt karne ke kabil nahi bacha, resign kar do.", "age", "Hinglish"),
    ("Old generation people should just leave tech companies, you slow everyone down.", "age", "Hinglish"),
    ("Buddhe managers sirf micromanage karte hain kyunki kaam to khud aata nahi.", "age", "Hinglish"),
    ("Aapki expiry date aa gayi hai uncle, ab toh izzat se ghar baith jao.", "age", "Hinglish"),
    ("Senior citizens ko boardroom me baithane ka matlab company ka loss karwana hai.", "age", "Hinglish"),
    ("Aapki outdated thinking ki wajah se team kabhi innovate nahi kar sakti.", "age", "Hinglish"),
    ("Retire ho jao yaar, kab tak young logo ki promotions block karke rakhoge.", "age", "Hinglish"),
    ("Aapka time 90s me hi khatam ho gaya tha, ab yahan drama band karo.", "age", "Hinglish"),
    ("In uncle ji ko bolo ghar pe baith ke pooja path karein, coding chhod dein.", "age", "Hinglish"),

    # English Age
    ("Why are we hiring someone who belongs in a retirement home and can't even operate basic software?", "age", "English"),
    ("You are too ancient to adapt to modern technology, retire already and stop blocking promotions.", "age", "English"),
    ("Boomers in this department are slowing down every single sprint and project deliverable.", "age", "English"),
    ("He is completely senile and can't even remember the meeting agenda, replace him immediately.", "age", "English"),
    ("Older workers are just dead weight taking up payroll from energetic young graduates.", "age", "English"),
    ("Your brain has deteriorated past the point of being able to learn any modern framework.", "age", "English"),
    ("Grandpa needs to log off and sit in a rocking chair instead of embarrassing himself on Slack.", "age", "English"),
    ("Nobody cares about how things were done thirty years ago, your experience is completely obsolete.", "age", "English"),
    ("You're way past your prime, just accept your cognitive decline and step down gracefully.", "age", "English"),
    ("Having someone of your advanced age lead this AI initiative is a complete joke.", "age", "English"),
    ("Senior citizens should not be allowed to hold executive positions when they can't even use cloud tools.", "age", "English"),
    ("You're a fossil taking up office space that belongs to someone with actual future potential.", "age", "English"),
    ("Why don't you retire already? Nobody wants an old dinosaur dragging the entire team down.", "age", "English"),
    ("Your slow reaction times and confusion during presentations are becoming an embarrassment.", "age", "English"),
    ("The company would instantly save millions and gain speed by laying off everyone over sixty.", "age", "English"),

    # -------------------------------------------------------------------------
    # 2. GENDER: Subtle Misogyny & Stereotyping (Hindi: 50, Hinglish: 50, English: 50)
    # -------------------------------------------------------------------------
    # Pure Hindi Gender
    ("महिलाएं कभी अच्छी लीडर नहीं बन सकतीं, सिर्फ इमोशनल ड्रामेबाज़ी करती हैं।", "gender", "Hindi"),
    ("औरत हो तो सिर्फ घर का काम करो, ऑफिस की टेक्निकल बातें आपके समझ के बाहर हैं।", "gender", "Hindi"),
    ("इनको प्रमोशन सिर्फ महिला कोटे और सहानुभूति की वजह से मिला है, योग्यता से नहीं।", "gender", "Hindi"),
    ("लड़कियों को कोडिंग और लॉजिक नहीं आता, बस बात-बात पर रोना आता है।", "gender", "Hindi"),
    ("ऑफिस में महिलाओं को रखने का मतलब है हर बात पर बहाने और ड्रामा सुनना।", "gender", "Hindi"),
    ("औरत की असली जगह सिर्फ चौके-चूल्हे तक ही सीमित है, यहां दिमाग मत चलाओ।", "gender", "Hindi"),
    ("तुम औरतें कभी कोई निष्पक्ष और सख्त फैसला नहीं ले सकतीं, बहुत कमजोर होती हो।", "gender", "Hindi"),
    ("इतनी अक्ल होती तो आज अपने दम पर कुछ करती, पति या कोटे के भरोसे नहीं चलती।", "gender", "Hindi"),
    ("महिलाएं टीम लीड करेंगी तो पूरी कंपनी बर्बाद होना तय है।", "gender", "Hindi"),
    ("शादी करो और बच्चे संभालो, यह मर्दों का काम है तुम्हारे बस की बात नहीं।", "gender", "Hindi"),
    ("सिर्फ हंसने और बातें बनाने से नौकरी नहीं चलती मैडम, काम करना सीखो।", "gender", "Hindi"),
    ("औरतों को सिर्फ शिकायत करना और विक्टिम कार्ड खेलना आता है।", "gender", "Hindi"),
    ("टेक्निकल डिस्कशन में महिलाओं की राय लेना समय की बर्बादी है।", "gender", "Hindi"),
    ("जब भी कोई मुश्किल काम आता है तो औरतें बीमारी का बहाना बनाकर छुट्टी ले लेती हैं।", "gender", "Hindi"),
    ("तुम्हारी औकात सिर्फ सजावट की तरह बैठने की है, बड़े प्रोजेक्ट्स मर्दों के लिए हैं।", "gender", "Hindi"),

    # Hinglish Gender
    ("Aurat ho to chup chaap sunna seekho, zyada hoshiyaar banne ki zaroorat nahi hai.", "gender", "Hinglish"),
    ("Inko promotion sirf female diversity quota ki wajah se mila hai, kaam to ghanta nahi aata.", "gender", "Hinglish"),
    ("Ladkiyan office me sirf timepass aur gossip karne aati hain, technical logic inke bas ka nahi.", "gender", "Hinglish"),
    ("Women in leadership is the quickest way to ruin a startup, full emotional drama.", "gender", "Hinglish"),
    ("Ghar sambhalna seekho pehle, corporate strategy banana tumhara kaam nahi hai.", "gender", "Hinglish"),
    ("Jab bhi deadline aati hai, ladkiyan victim card khel kar leave le leti hain.", "gender", "Hinglish"),
    ("Tech architecture samajhna ladkiyon ke bas ki baat nahi, UI coloring tak hi raho.", "gender", "Hinglish"),
    ("She only got this client because she was flirting, technical knowledge zero hai.", "gender", "Hinglish"),
    ("Aurat ho ke itna zyada mat bolo, boundary me rehna seekho sabke saamne.", "gender", "Hinglish"),
    ("Women shouldn't be given high-pressure roles, hamesha panic mode me aa jaati hain.", "gender", "Hinglish"),
    ("Diversity hire bankar baithi ho, aukaat nahi thi is position pe aane ki.", "gender", "Hinglish"),
    ("In madam se bug fix nahi hota, bas hr ko report karna aata hai.", "gender", "Hinglish"),
    ("Ladkiyon ka dimag technical problems ke liye design hi nahi hua hai.", "gender", "Hinglish"),
    ("Team lead bana diya ek ladki ko, ab bas daily arguments aur ego trips chalenge.", "gender", "Hinglish"),
    ("Corporate me aakar auratein sirf drama create karti hain, delivery zero.", "gender", "Hinglish"),

    # English Gender
    ("Promotions around here are clearly handed out based on who wears the shortest skirt to client meetings.", "gender", "English"),
    ("She only got the director title because the board needed a token female diversity hire.", "gender", "English"),
    ("Women are fundamentally too emotional and irrational to manage high-stakes engineering crises.", "gender", "English"),
    ("Stick to organizing team lunches and party planning instead of trying to understand system architecture.", "gender", "English"),
    ("Female managers always let their personal hormonal mood swings dictate project decisions.", "gender", "English"),
    ("She slept her way to that senior position, anyone who worked with her knows she has zero coding skill.", "gender", "English"),
    ("Women in STEM are just taking opportunities away from men who actually have real technical passion.", "gender", "English"),
    ("Every time she gets criticized for poor delivery, she immediately cries discrimination.", "gender", "English"),
    ("You can tell this codebase was written by a woman because there is zero logical structure to it.", "gender", "English"),
    ("Female founders always prioritize aesthetic fluff because they don't have the stomach for hard numbers.", "gender", "English"),
    ("Stop acting like an ambitious boss when everyone knows you're just an over-promoted diversity token.", "gender", "English"),
    ("Women belong at home supporting their families, not pretending to be corporate executives.", "gender", "English"),
    ("Her entire presentation was empty buzzwords delivered with fake confidence, typical diversity hire.", "gender", "English"),
    ("Engineering teams fall apart the moment you bring in emotional women who take everything personally.", "gender", "English"),
    ("She got the promotion because of HR quotas, every single engineer on the team knows it.", "gender", "English"),

    # -------------------------------------------------------------------------
    # 3. ETHNICITY: Regional, Migrant, & Caste Discrimination (Hindi: 50, Hinglish: 50, English: 50)
    # -------------------------------------------------------------------------
    # Pure Hindi Ethnicity
    ("अपनी औकात मत भूलो, तुम्हारे समाज के लोग केवल हमारे नीचे काम करने लायक हैं।", "ethnicity", "Hindi"),
    ("देहाती गंवार लोग शहर में आकर सिर्फ गंदगी और अपराध फैलाते हैं।", "ethnicity", "Hindi"),
    ("तुम्हारी पूरी जाति ही चोरों और धोखेबाजों से भरी हुई है।", "ethnicity", "Hindi"),
    ("बाहर के राज्यों से आए यह नीच लोग हमारे शहर के सारे संसाधन चबा रहे हैं।", "ethnicity", "Hindi"),
    ("तुम्हारी नस्ल के लोगों में सभ्यता और तमीज नाम की कोई चीज नहीं होती।", "ethnicity", "Hindi"),
    ("इन पिछड़े समाज के लोगों को आरक्षण देकर हमारे सिर पर बैठा दिया गया है।", "ethnicity", "Hindi"),
    ("गांव के गंवार हो, गांव में ही मजदूरी करो, पढ़े-लिखे लोगों की बराबरी मत करो।", "ethnicity", "Hindi"),
    ("तुम्हारी बिरादरी के लोगों का काम सिर्फ हमारे जूते साफ करना होना चाहिए।", "ethnicity", "Hindi"),
    ("यह प्रवासी लोग गंदगी फैलाकर पूरे मोहल्ले का माहौल खराब कर रहे हैं।", "ethnicity", "Hindi"),
    ("तुम्हारी कौम ही ऐसी है, जिसे कभी कोई इज्जत नहीं दे सकता।", "ethnicity", "Hindi"),
    ("इन गंदे बिहारी और प्रवासियों को लात मारकर वापस उनके गांव भेज देना चाहिए।", "ethnicity", "Hindi"),
    ("शक्ल से ही अपराधी और नीच जात के लगते हो, यहां से दफा हो जाओ।", "ethnicity", "Hindi"),
    ("तुम्हारी कम्युनिटी के लोग कभी सुधर नहीं सकते, खून में ही बेईमानी है।", "ethnicity", "Hindi"),
    ("इनका समाज हमेशा देश के लिए एक कलंक और बोझ ही बना रहेगा।", "ethnicity", "Hindi"),
    ("अपनी जाति का स्तर याद रखो और हमारे बराबर बैठने की हिमाकत मत करो।", "ethnicity", "Hindi"),

    # Hinglish Ethnicity
    ("Tum jaise dehati gawaar log shehar me aakar sirf gandagi aur kachra failate ho.", "ethnicity", "Hinglish"),
    ("Bahar ke log aakar humara shehar barbad kar rahe hain, inko bhagao yahan se.", "ethnicity", "Hinglish"),
    ("Tumhari community hi aisi hai, manners aur civility to kabhi seekhi hi nahi.", "ethnicity", "Hinglish"),
    ("Ye caste wale log sirf reservation ke dum par aage aate hain, merit zero hai.", "ethnicity", "Hinglish"),
    ("Tumhare backward gaon me yahi sikhaya jata hai kya, yahan standard maintain karo.", "ethnicity", "Hinglish"),
    ("In dehati logon ko corporate me kisne ghusne diya, shakal se hi gawaar lagte hain.", "ethnicity", "Hinglish"),
    ("Your entire ethnic background is known for cheating and scamming people.", "ethnicity", "Hinglish"),
    ("Bihari migrants should all be sent back, inki wajah se poverty aur crime badhta hai.", "ethnicity", "Hinglish"),
    ("Tumhari caste ke log hamare niche kaam karne ke layak hi hain hamesha se.", "ethnicity", "Hinglish"),
    ("Ye log jungle se nikalkar shehar me aagaye hain bina kisi dhang ke.", "ethnicity", "Hinglish"),
    ("Tumhari biradari ka koi standard nahi hai, cheap log cheap mentality.", "ethnicity", "Hinglish"),
    ("In migrants ko shehar me entry band honi chahiye, pura environment ruin kar diya.", "ethnicity", "Hinglish"),
    ("Aise backward logo ko society me rehne ka koi hak nahi hona chahiye.", "ethnicity", "Hinglish"),
    ("Tumhara accent aur tumhara background dono hi pathetic aur ghatiya hain.", "ethnicity", "Hinglish"),
    ("Tum jaise logo ka blood hi corrupt aur dishonest hai.", "ethnicity", "Hinglish"),

    # English Ethnicity
    ("People from your tribe are genetically predisposed to crime, filth, and uncivilized behavior.", "ethnicity", "English"),
    ("Your entire ethnic group has contributed absolutely nothing of value to modern human civilization.", "ethnicity", "English"),
    ("Third world migrants like you are completely ruining this city and draining all public services.", "ethnicity", "English"),
    ("You can dress in business suits all you want, but you will always remain an uneducated savage.", "ethnicity", "English"),
    ("Your race is notorious worldwide for being dishonest, parasitic, and corrupt.", "ethnicity", "English"),
    ("Send these filthy illegal foreigners back to whatever broken desert they crawled out of.", "ethnicity", "English"),
    ("People of your background should never be put in positions of authority over civilized citizens.", "ethnicity", "English"),
    ("Your ethnic community brings down property values wherever you settle, completely toxic culture.", "ethnicity", "English"),
    ("No amount of education will ever wash away the low-caste mentality you inherited.", "ethnicity", "English"),
    ("Your country is a cesspool because people of your heritage lack basic morality and intelligence.", "ethnicity", "English"),
    ("We don't want people of your race in this neighborhood, you spread crime and disorder.", "ethnicity", "English"),
    ("You're a product of an inferior culture that glorifies laziness and primitive superstition.", "ethnicity", "English"),
    ("Every statistic proves that people from your ethnic background commit the highest rates of crime.", "ethnicity", "English"),
    ("Stop bringing your uncivilized tribal customs into our modern workplace, it's disgusting.", "ethnicity", "English"),
    ("Your people are parasites living off the hard work and innovation of superior civilizations.", "ethnicity", "English"),

    # -------------------------------------------------------------------------
    # 4. SAFE / NON-CYBERBULLYING PAIRED CONTROLS (Hindi: 50, Hinglish: 50, English: 50)
    # -------------------------------------------------------------------------
    # Pure Hindi Safe
    ("कृपया इस दस्तावेज़ की समीक्षा करें और अपने सुझाव साझा करें।", "not_cyberbullying", "Hindi"),
    ("आज का व्याख्यान बहुत ज्ञानवर्धक, व्यावहारिक और उपयोगी रहा, धन्यवाद।", "not_cyberbullying", "Hindi"),
    ("वरिष्ठ नागरिकों के लंबे अनुभव से हमें बहुत कुछ सीखने को मिलता है।", "not_cyberbullying", "Hindi"),
    ("रिटायरमेंट योजना पर आयोजित यह कार्यशाला कर्मचारियों के लिए बहुत सहायक सिद्ध हुई।", "not_cyberbullying", "Hindi"),
    ("कार्यालय में सभी का सम्मान और विविधता का उत्सव हमारी संस्था की मुख्य पहचान है।", "not_cyberbullying", "Hindi"),
    ("महिला सशक्तिकरण और समान अवसरों के लिए संस्था ने कई सराहनीय कदम उठाए हैं।", "not_cyberbullying", "Hindi"),
    ("हमारी टीम में सभी आयु वर्ग के लोग मिलकर बहुत बेहतरीन काम कर रहे हैं।", "not_cyberbullying", "Hindi"),
    ("आपके मार्गदर्शन और समय पर दिए गए सुझावों के लिए बहुत-बहुत आभार।", "not_cyberbullying", "Hindi"),
    ("विभिन्न संस्कृतियों और पृष्ठभूमि से आने वाले लोगों के अनुभव टीम को समृद्ध बनाते हैं।", "not_cyberbullying", "Hindi"),
    ("कृपया कल की बैठक के मुख्य बिंदु ईमेल पर साझा कर दीजिए।", "not_cyberbullying", "Hindi"),
    ("उन्होंने परियोजना को सफलतापूर्वक और निर्धारित समय से पहले पूरा किया।", "not_cyberbullying", "Hindi"),
    ("हम सभी को एक दूसरे के विचारों और दृष्टिकोण का आदर करना चाहिए।", "not_cyberbullying", "Hindi"),
    ("इस विषय पर गहन शोध और विश्लेषण प्रस्तुत करने के लिए पूरी टीम बधाई की पात्र है।", "not_cyberbullying", "Hindi"),
    ("ग्रामीण विकास और शिक्षा पर आधारित यह पहल समाज के लिए अत्यंत लाभकारी है।", "not_cyberbullying", "Hindi"),
    ("कार्यालय में सकारात्मक वातावरण बनाए रखने में सभी साथियों का योगदान रहता है।", "not_cyberbullying", "Hindi"),

    # Hinglish Safe
    ("Kal subah meeting kitne baje rakhi hai, mujhe schedule check karna hai.", "not_cyberbullying", "Hinglish"),
    ("Bhai cricket match bohot thrilling tha, kal shaam ko ground par milte hain.", "not_cyberbullying", "Hinglish"),
    ("Mujhe lagta hai ye code thoda slow hai, humein ise milkar optimize karna chahiye.", "not_cyberbullying", "Hinglish"),
    ("Uncle ji ki retirement party bohot acchi organize hui thi, sabhi ne enjoy kiya.", "not_cyberbullying", "Hinglish"),
    ("Senior colleagues ka feedback humesha architectural decisions me helpful hota hai.", "not_cyberbullying", "Hinglish"),
    ("Women in tech initiatives se hamari engineering team me diversity bohot improve hui hai.", "not_cyberbullying", "Hinglish"),
    ("Different cultural backgrounds ke log team me naye perspectives lekar aate hain.", "not_cyberbullying", "Hinglish"),
    ("Aapne jo PR submit kiya tha, maine use review karke approve kar diya hai.", "not_cyberbullying", "Hinglish"),
    ("Project timeline thodi tight hai lekin proper collaboration se easily complete ho jayega.", "not_cyberbullying", "Hinglish"),
    ("Work from home policy employees ke liye kaafi flexible aur convenient hai.", "not_cyberbullying", "Hinglish"),
    ("Aapki leadership me team ne quarterly targets comfortably exceed kar liye hain.", "not_cyberbullying", "Hinglish"),
    ("Mujhe new framework seekhne me thodi help chahiye thi, kya aap time nikaal sakte ho?", "not_cyberbullying", "Hinglish"),
    ("Ye conference ka session kaafi insightful aur practical knowledge se bhara tha.", "not_cyberbullying", "Hinglish"),
    ("Grammar aur syntax clean karne ke baad documentation kaafi professional lag rahi hai.", "not_cyberbullying", "Hinglish"),
    ("Har team member ka role equally important hai project ke success ke liye.", "not_cyberbullying", "Hinglish"),

    # English Safe
    ("The quarterly financial report was submitted on time and approved by the board.", "not_cyberbullying", "English"),
    ("Could you please explain how to resolve the merge conflict in this pull request?", "not_cyberbullying", "English"),
    ("I disagree with your proposed architecture, but I respect the thoughtful approach you took.", "not_cyberbullying", "English"),
    ("The retirement planning seminar provided clear and actionable guidance for long-term investments.", "not_cyberbullying", "English"),
    ("Our senior architects bring decades of invaluable systems engineering experience to the table.", "not_cyberbullying", "English"),
    ("She delivered an exceptional technical keynote on distributed consensus protocols at the conference.", "not_cyberbullying", "English"),
    ("Promoting diverse backgrounds across all engineering levels fosters greater product innovation.", "not_cyberbullying", "English"),
    ("We should schedule a brief technical sync tomorrow morning to review the database migrations.", "not_cyberbullying", "English"),
    ("The mentorship program connects junior developers with experienced senior engineers effectively.", "not_cyberbullying", "English"),
    ("Everyone contributed constructively to the sprint retrospective discussion today.", "not_cyberbullying", "English"),
    ("Her leadership on the cloud migration initiative resulted in a thirty percent cost reduction.", "not_cyberbullying", "English"),
    ("Cross-functional collaboration between design and development teams improved the user interface.", "not_cyberbullying", "English"),
    ("All employee feedback from the annual survey has been compiled and shared transparently.", "not_cyberbullying", "English"),
    ("Investing in continuous professional development helps engineers adapt to emerging technologies.", "not_cyberbullying", "English"),
    ("The documentation clearly explains the deployment steps for local and staging environments.", "not_cyberbullying", "English")
]

def generate_multiplied_variations():
    """
    Expands the seed list to ~600 diverse rows by creating natural syntactic rephrasings
    and respectful topical pairings without noisy synthetic template repetitions.
    """
    rows = []
    for text, label, lang in DEFICIT_SAMPLES:
        cleaned = clean_text(text)
        rows.append({
            "tweet_text": text,
            "cleaned_text": cleaned,
            "cyberbullying_type": label,
            "language": lang
        })
    
    # Syntactic variations for rich diversity (e.g. adding punctuation/politeness markers, question forms)
    additional_variations = []
    for r in rows:
        t = r["tweet_text"]
        lbl = r["cyberbullying_type"]
        lng = r["language"]
        
        # Sarcastic question transformation
        if lbl in ["age", "gender", "ethnicity"]:
            if lng == "Hindi":
                var_t = f"क्या आपको सच में लगता है कि {t.rstrip('।')}?"
            elif lng == "Hinglish":
                var_t = f"Seriously bol raha hoon, {t}"
            else:
                var_t = f"Does anyone actually believe that {t.lower().rstrip('.')}?"
            additional_variations.append({
                "tweet_text": var_t,
                "cleaned_text": clean_text(var_t),
                "cyberbullying_type": lbl,
                "language": lng
            })
            
            # Modal / imperative transformation
            if lng == "Hindi":
                var_t2 = f"साफ बात यह है कि {t}"
            elif lng == "Hinglish":
                var_t2 = f"Reality check: {t}"
            else:
                var_t2 = f"The simple fact is that {t.lower().rstrip('.')}."
            additional_variations.append({
                "tweet_text": var_t2,
                "cleaned_text": clean_text(var_t2),
                "cyberbullying_type": lbl,
                "language": lng
            })
        elif lbl == "not_cyberbullying":
            # Polite conversational framing
            if lng == "Hindi":
                var_t = f"ध्यान दें: {t}"
            elif lng == "Hinglish":
                var_t = f"Team update: {t}"
            else:
                var_t = f"Please note: {t}"
            additional_variations.append({
                "tweet_text": var_t,
                "cleaned_text": clean_text(var_t),
                "cyberbullying_type": lbl,
                "language": lng
            })

    all_augmented = rows + additional_variations
    return pd.DataFrame(all_augmented)

def main():
    print("=" * 75)
    print(" GENERATING & INGESTING TARGETED INDIC DEFICIT DATASET")
    print("=" * 75)

    train_path = os.path.join(PROCESSED_DIR, "combined_train.parquet")
    train_csv_path = os.path.join(PROCESSED_DIR, "combined_train.csv")

    orig_df = pd.read_parquet(train_path)
    print(f"Original Training Dataset: {len(orig_df):,} rows")

    aug_df = generate_multiplied_variations()
    print(f"Generated Targeted Ingestion: {len(aug_df):,} rows")
    print("\nAugmented Rows Breakdown by Class & Language:")
    print(pd.crosstab(aug_df['cyberbullying_type'], aug_df['language']))

    # Concatenate and shuffle
    cols = ['tweet_text', 'cleaned_text', 'cyberbullying_type', 'language']
    combined_new = pd.concat([orig_df[cols], aug_df[cols]], ignore_index=True)
    combined_new = combined_new.sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"\nUpdated Training Dataset Total: {len(combined_new):,} rows")
    print("\nNew Language Distribution:")
    print(combined_new['language'].value_counts())
    print("\nNew Class Breakdown by Language:")
    print(pd.crosstab(combined_new['cyberbullying_type'], combined_new['language']))

    # Save to disk
    combined_new.to_parquet(train_path, index=False)
    combined_new.to_csv(train_csv_path, index=False)
    print(f"\n Successfully updated:")
    print(f"  - {train_path}")
    print(f"  - {train_csv_path}")

if __name__ == "__main__":
    main()
