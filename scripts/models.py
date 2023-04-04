import torch
import pickle
from torch import nn
from transformers import AutoModel
from metrics import display_metrics
from data import load_data, load_data_bert, device


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


class BertFinetune(nn.Module):

    def __init__(self):
        super(BertFinetune, self).__init__()
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.out_layer = nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        _, features = self.bert(
            ids, token_type_ids=token_type_ids,
            attention_mask=mask, return_dict=False
        )
        output = self.out_layer(features)
        return output


def get_vectorizer():
    while True:
        print("Choose vectorizer (bow, tfidf):", end=" ")
        vectorizer_choice = str(input())
        if vectorizer_choice in ["bow", "tfidf"]:
            break
        else:
            print("Please provide a valid vectorizer!")

    vectorizer_path = "models/"

    if vectorizer_choice == "bow":
        vectorizer_path += "bag-of-words/vectorizer/bow_vectorizer"
    else:
        vectorizer_path += "tf-idf/vectorizer/tfidf_vectorizer"

    vectorizer = pickle.load(open(vectorizer_path, "rb"))

    return vectorizer_choice, vectorizer


def get_model(model_name, vectorizer_name):
    model_path = "models/"

    if vectorizer_name == "bow":
        model_path += f"bag-of-words/models/bow_{model_name}_model/model"
    else:
        model_path += f"tf-idf/models/tfidf_{model_name}_model/model"

    model = pickle.load(open(model_path, "rb"))

    return model


def linear_regression():
    vectorizer_choice, vectorizer = get_vectorizer()
    model = get_model("lr", vectorizer_choice)
    X_train, X_test, y_train, y_test = load_data(vectorizer)
    print("Getting predictions...")
    pred = model.predict(X_test)
    display_metrics(pred, y_test)


def svm():
    vectorizer_choice, vectorizer = get_vectorizer()
    model = get_model("svm", vectorizer_choice)
    X_train, X_test, y_train, y_test = load_data(vectorizer)
    print("Getting predictions...")
    pred = model.predict(X_test)
    display_metrics(pred, y_test)


def get_variant():
    while True:
        print("Choose variant (vanilla(v), finetuned(f)):", end=" ")
        variant = str(input())
        if variant in ["v", "f"]:
            break
        else:
            print("Please provide a valid variant!")

    return variant


def bert():
    variant = get_variant()
    train_loader, test_loader = load_data_bert()

    weights_path = "models/"

    if variant == "v":
        model = BertVanilla().to(device)
        weights_path += "bert-vanilla.pt"
        weights = torch.load(weights_path)
    if variant == "f":
        model = BertFinetune().to(device)
        weights_path += "bert-finetune.pt"
        weights = torch.load(weights_path)

    model.load_state_dict(weights)

    model.eval()

    targets = []
    outputs = []

    with torch.no_grad():
        for batch in test_loader:
            ids = batch['ids'].to(device, dtype=torch.long)
            mask = batch['mask'].to(device, dtype=torch.long)
            token_type_ids = batch['token_type_ids'].to(
                device, dtype=torch.long)
            batch_targets = batch['targets'].to(device, dtype=torch.float)

            batch_outputs = model(ids, mask, token_type_ids)
            batch_outputs = batch_outputs.view(-1)

            targets.extend(batch_targets.cpu().numpy().tolist())
            outputs.extend(batch_outputs.cpu().numpy().tolist())

    display_metrics(targets, outputs)
