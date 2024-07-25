# Importing pipeline from transformers package
from transformers import pipeline

# Setting up a sentiment analysis classifier
classifier = pipeline(task = "sentiment-analysis",
                      model = "distilbert-base-uncased-finetuned-sst-2-english")
print(classifier)

# Sample text
sample_text = ["It is unfair",
               "I think it is a good idea."]

print(classifier(sample_text))