# Import libraries
import re
import torch
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm
from lime.lime_text import LimeTextExplainer
from transformers import AutoModel, AutoTokenizer


# Define model
class BertVanilla(nn.Module):

    def __init__(self):
        super(BertVanilla, self).__init__()
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = False
        self.out_layer = nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        _, features = self.bert(
            ids, token_type_ids=token_type_ids,
            attention_mask=mask, return_dict=False
        )
        output = self.out_layer(features)
        return output


# Intialize compute device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize model
model = BertVanilla().to(device)

# Load model weights
weights = torch.load("models/bert-vanilla.pt")
model.load_state_dict(weights)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')


def clean_text_bert(text):
    text = text.lower()  # Convert the text into lowercase
    text = text.replace('</br>', '')  # Remove </br>
    text = text.replace('\n', '')  # Remove \n

    # Remove quotes
    text = re.sub(r"\'", "", text)
    text = re.sub(r"\"", "", text)

    text = re.sub(r"[^\w]", " ", text)  # Remove all symbols

    text = re.sub(r'[ ]{2,}', ' ', text)  # Remove extra spaces
    text = re.sub(r'[ \t]+$', '', text)  # Remove trailing white spaces

    return text


def pipeline(text, tokenizer, max_len, model):
    tokens = tokenizer.encode_plus(
        text,
        truncation=True,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        return_token_type_ids=True
    )

    ids = torch.tensor(tokens['input_ids']).to(device).unsqueeze(0)
    mask = torch.tensor(tokens['attention_mask']).to(device).unsqueeze(0)
    token_type_ids = torch.tensor(tokens["token_type_ids"]).to(device)

    with torch.no_grad():
        model.eval()
        output = model(ids, mask, token_type_ids)

    return output.item()


def read_data():
    df = pd.read_csv("data/arg_quality_rank_30k.csv")
    df["argument"] = df["argument"].apply(clean_text_bert)
    df["topic"] = df["topic"].apply(clean_text_bert)
    max_len = np.max([len(x) for x in df["argument"]])
    return df["argument"].tolist(), df["topic"].tolist(), max_len


arguments, topics, max_len = read_data()

# Generate random topic and argument
idx = np.random.randint(0, len(arguments))
topic = topics[idx]
text = arguments[idx]
print(f"TOPIC: {topic}")
print(f"ARGUMENT: {text}")

val = pipeline(text, tokenizer, max_len, model)
print(f"QUALITY: {val:>.4f}")


def lime_predict(text):
    if isinstance(text, list):
        output = [pipeline(ele, tokenizer, max_len, model)
                  for ele in tqdm(text)]
        output = [[1-ele, ele] for ele in output]
        output = np.array(output)
    else:
        output = pipeline(text, tokenizer, max_len, model)
    return output


# Explain model
explainer = LimeTextExplainer()
explanation = explainer.explain_instance(
    text, lime_predict, num_features=10, num_samples=5000)

# Write explanation to html file
with open("explainability/explanation.html", "w") as file:
    file.write(explanation.as_html())

# Print tokens and their weights
token_df = pd.DataFrame()
token_df["TOKEN"] = [x for (x, y) in explanation.as_list()]
token_df["WEIGHT"] = [y for (x, y) in explanation.as_list()]
print(token_df)
