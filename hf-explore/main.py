from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

categories = ["food and drink", "music", "cultural", "fitness", "tech", "fashion", "charity"]

events = [
    "Culture Night: a celebration of South Asian food, dance, and live music",
    "Spring 5K fun run raising money for the local food bank",
    "Startup pitch night with free coffee for attendees",
]

for e in events:
    result = classifier(e, categories, multi_label=True)
    print(e)
    for label, score in zip(result["labels"][:3], result["scores"][:3]):
        print(f"  {label}: {score:.0%}")
    print()