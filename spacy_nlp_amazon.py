import spacy
from textblob import TextBlob

# ---------- Step 1: Load spaCy pre-trained model ----------
# Make sure to install dependencies first:
# python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

# ---------- Step 2: Sample Amazon Product Reviews ----------
reviews = [
    "I absolutely love this Samsung phone. The battery life is amazing and the camera quality is outstanding!",
    "This Apple MacBook is overpriced and the keyboard is terrible. I expected better from Apple.",
    "The Sony headphones have great sound quality but the build feels cheap.",
    "Amazon Echo is a fantastic product! Voice recognition is top-notch and very useful at home.",
    "I am very disappointed with this Dell laptop. It crashes frequently and the performance is awful."
]

# ---------- Step 3: Define rule-based sentiment analyzer ----------
positive_words = ["love", "amazing", "outstanding", "great", "fantastic", "top-notch", "useful"]
negative_words = ["overpriced", "terrible", "cheap", "disappointed", "awful", "crashes", "bad"]

def analyze_sentiment(text):
    text_lower = text.lower()
    score = 0
    for word in positive_words:
        if word in text_lower:
            score += 1
    for word in negative_words:
        if word in text_lower:
            score -= 1
    return "Positive" if score > 0 else "Negative" if score < 0 else "Neutral"

# ---------- Step 4: Process reviews & extract entities ----------
for review in reviews:
    print("Review:", review)
    doc = nlp(review)

    # Extract product and brand entities
    print("Named Entities (Product/Brand Extracted):")
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT"]:  # Using ORG to detect brands like Apple, Samsung
            print(f" - {ent.text} ({ent.label_})")

    # Analyze sentiment
    sentiment = analyze_sentiment(review)
    print("Sentiment:", sentiment)
    print("-" * 50)
