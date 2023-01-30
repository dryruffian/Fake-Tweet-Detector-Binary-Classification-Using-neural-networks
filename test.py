import json
import torch
from fastapi import FastAPI
from torch import Tensor
from transformers import BertForSequenceClassification, BertTokenizer
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
state_dict = torch.load('models/fake_tweet2.ckpt')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.load_state_dict(state_dict)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class model_input(BaseModel):
    tweet: str


@app.post("/predict")
def predict(input_parameter: model_input):
    input_data = input_parameter.json()
    dict_parameter = json.loads(input_data)
    tweet = dict_parameter["tweet"]

    encoded_input = tokenizer.encode_plus(tweet, add_special_tokens=True, return_attention_mask=True)
    input_ids, attention_mask = encoded_input["input_ids"], encoded_input["attention_mask"]
    input_tensor: Tensor = torch.tensor([input_ids])
    attention_tensor = torch.tensor([attention_mask])
    model.eval()

    with torch.no_grad():
        outputs = model(input_tensor, attention_mask=attention_tensor)
        logits = outputs[0]
        predictions = torch.softmax(logits, dim=1)

    predicted_class = torch.argmax(predictions).item()
    predicted_probability = predictions[0][predicted_class].item()

    if predicted_class == 1:
        response_text = 0
    else:
        response_text = 1
    response_data = {"prediction": response_text, "probability": predicted_probability}
    print(response_data)

    return response_data
# predict({"tweet": "This tweet is fake news"})
