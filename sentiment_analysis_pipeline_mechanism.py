# Pipeline machanism (three steps)
# - preprocessing
# - passing the inputs through the model
# - postprocessing

# Importing the packages
from transformers import AutoTokenizer, AutoModelForSequenceClassification  # to perform classification task
import torch

# Setting the checkpoint
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"


# 1. Tokenization
# Tokenizers split inputs into words, subwords, or symbols that are called tokens
# Each token is mapped to an integer, and additional inputs that might be useful to the model are added
# initializing the tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Raw inputs
raw_inputs = ["It is unfair",
               "I think it is a good idea."]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors='pt')  # 'pt' standards for pytorch
print(inputs)
print()


# 2. Going through the model
# Outputs of tokenizer can be processed through the pretrained model.
# Performing a classification task is done by model heads, which are specific layers that tweak the transformer architecture.
# Initializing the model - using AutoModelForSequenceClassification instead of AutoModel
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)         # the double asterisk ensures that the inputs are stored as a dictionary within the function
print(outputs.logits.shape)
print(outputs.logits)
print()



# 3. Postprocessing the output
# Tranformer models output logits - raw, unnormalized scores outputted by the last layer of the model
# passed through a softmax function that can convert this tensor output into a probability distribution of outcomes
# Converting the tensor output to a probability distribution
prediction = torch.nn.functional.softmax(outputs.logits, dim=-1)

print(prediction)            # matrix that corresponds to the probability that a sentence belongs to either class 
print(model.config.id2label) # class labels for the classification