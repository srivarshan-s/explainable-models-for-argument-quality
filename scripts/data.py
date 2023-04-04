import re
import nltk
import spacy
import torch
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from gensim.utils import simple_preprocess

nltk.download('stopwords')

stop_words = stopwords.words('english')
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
allowed_postags = ["NOUN", "ADJ", "VERB"]

device = "cuda" if torch.cuda.is_available() else "cpu"


class ModelDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.max_len = max_len
        self.text = df["argument"].values
        self.tokenizer = tokenizer
        self.targets = df["WA"].values

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        text = self.text[index]
        inputs = self.tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long).to(device),
            'mask': torch.tensor(mask, dtype=torch.long).to(device),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long).to(device),
            'targets': torch.tensor(self.targets[index], dtype=torch.float).to(device)
        }


def clean_text(text):
    text = text.lower()  # Convert the text into lowercase
    text = text.replace('</br>', '')  # Remove </br>
    text = text.replace('\n', '')  # Remove \n

    # Remove quotes
    text = re.sub(r"\'", "", text)
    text = re.sub(r"\"", "", text)

    text = re.sub(r"[^\w]", " ", text)  # Remove all symbols

    text = re.sub(r'[ ]{2,}', ' ', text)  # Remove extra spaces
    text = re.sub(r'[ \t]+$', '', text)  # Remove trailing white spaces

    text = simple_preprocess(str(text), deacc=True)  # Tokenize the texts

    # Remove stopwords
    tokens = []
    for token in text:
        if token not in stop_words:
            tokens.append(token)
    text = tokens

    # Lemmatize the tokens
    text = " ".join(text)
    text = nlp(text)
    lemmatized_tokens = []
    for token in text:
        if token.pos_ in allowed_postags:
            lemmatized_tokens.append(token.lemma_)
    text = lemmatized_tokens

    text = " ".join(text)
    return text


def load_data(vectorizer):
    print("Loading data...")

    df = pd.read_csv("data/arg_quality_rank_30k.csv")

    df_train = df[df["set"] != "test"].reset_index(drop=True)
    df_train = df_train.drop(["set"], axis=1)

    df_test = df[df["set"] == "test"].reset_index(drop=True)
    df_test = df_test.drop(["set"], axis=1)

    df_train["argument"] = df_train["argument"].apply(clean_text)
    df_test["argument"] = df_test["argument"].apply(clean_text)

    X_train = vectorizer.transform(df_train["argument"]).toarray()
    X_test = vectorizer.transform(df_test["argument"]).toarray()

    y_train = np.array(df_train["WA"])
    y_test = np.array(df_test["WA"])

    return (X_train, X_test, y_train, y_test)


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


def load_data_bert():
    print("Loading data...")

    df = pd.read_csv("data/arg_quality_rank_30k.csv")

    df_train = df[df["set"] != "test"].reset_index(drop=True)
    df_train = df_train.drop(["set"], axis=1)

    df_test = df[df["set"] == "test"].reset_index(drop=True)
    df_test = df_test.drop(["set"], axis=1)

    df_train["argument"] = df_train["argument"].apply(clean_text_bert)
    df_test["argument"] = df_test["argument"].apply(clean_text_bert)

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    MAX_LEN = np.max([
        np.max([len(x) for x in df_train["argument"]]),
        np.max([len(x) for x in df_test["argument"]])
    ])

    BATCH_SIZE = 32

    train_data = ModelDataset(df_train, tokenizer, MAX_LEN)
    test_data = ModelDataset(df_test, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    return train_loader, test_loader
