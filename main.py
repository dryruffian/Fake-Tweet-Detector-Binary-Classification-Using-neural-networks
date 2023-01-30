import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import preprocessor as p

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

from tqdm import trange
import pandas as pd
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)

MAX_LEN = 128

df = pd.read_csv("data/Constraint_Train.csv")
val_df = pd.read_csv("data/Constraint_Val.csv")
test_df = pd.read_csv("data/Constraint_Test.csv")

wordnet_lemmatizer = WordNetLemmatizer()
porter_stemmer = PorterStemmer()

p.set_options(p.OPT.URL, p.OPT.EMOJI)


def preprocess(row):
    text = row['tweet']
    text = p.clean(text)
    return text


df['tweet'] = df.apply(lambda x: preprocess(x, wordnet_lemmatizer, porter_stemmer), 1)
val_df['tweet'] = val_df.apply(lambda x: preprocess(x, wordnet_lemmatizer, porter_stemmer), 1)
test_df['tweet'] = test_df.apply(lambda x: preprocess(x, wordnet_lemmatizer, porter_stemmer), 1)


def map_label(row):
    return 0 if row['label'] == 'real' else 1


df['label_encoded'] = df.apply(lambda x: map_label(x), 1)
val_df['label_encoded'] = val_df.apply(lambda x: map_label(x), 1)

train_sentences = df.tweet.values
val_sentences = val_df.tweet.values
test_sentences = test_df.tweet.values

train_labels = df.label_encoded.values
val_labels = val_df.label_encoded.values

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


def Encode_TextWithAttention(sentence, tokenizer, maxlen, padding_type='max_length', attention_mask_flag=True):
    encoded_dict = tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=maxlen, truncation=True,
                                         padding=padding_type, return_attention_mask=attention_mask_flag)
    return encoded_dict['input_ids'], encoded_dict['attention_mask']


def Encode_TextWithoutAttention(sentence, tokenizer, maxlen, padding_type='max_length', attention_mask_flag=False):
    encoded_dict = tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=maxlen, truncation=True,
                                         padding=padding_type, return_attention_mask=attention_mask_flag)
    return encoded_dict['input_ids']


def get_TokenizedTextWithAttentionMask(sentenceList, tokenizer):
    token_ids_list, attention_mask_list = [], []
    for sentence in sentenceList:
        token_ids, attention_mask = Encode_TextWithAttention(sentence, tokenizer, MAX_LEN)
        token_ids_list.append(token_ids)
        attention_mask_list.append(attention_mask)
    return token_ids_list, attention_mask_list


def get_TokenizedText(sentenceList, tokenizer):
    token_ids_list = []
    for sentence in sentenceList:
        token_ids = Encode_TextWithoutAttention(sentence, tokenizer, MAX_LEN)
        token_ids_list.append(token_ids)
    return token_ids_list


train_token_ids, train_attention_masks = torch.tensor(get_TokenizedTextWithAttentionMask(train_sentences, tokenizer))
val_token_ids, val_attention_masks = torch.tensor(get_TokenizedTextWithAttentionMask(val_sentences, tokenizer))
test_token_ids, test_attention_masks = torch.tensor(get_TokenizedTextWithAttentionMask(test_sentences, tokenizer))

train_labels = torch.tensor(train_labels)
val_labels = torch.tensor(val_labels)

batch_size = 16

train_data = TensorDataset(train_token_ids, train_attention_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(val_token_ids, val_attention_masks, val_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

test_data = TensorDataset(test_token_ids, test_attention_masks)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).cuda()

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)

train_loss_set = []
best_val_accuracy = 0.90
directory_path = ''
epochs = 30

for _ in trange(epochs, desc="Epoch"):
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0

    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        optimizer.zero_grad()
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs[0]
        logits = outputs[1]
        train_loss_set.append(loss.item())
        loss.backward()
        optimizer.step()

        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1

    print("Train loss: {}".format(tr_loss / nb_tr_steps))

    model.eval()

    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = output[0]

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        tmp_eval_accuracy = flat_accuracy(logits, label_ids)

        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

    print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))
    Validation_Accuracy = (eval_accuracy / nb_eval_steps)
    if (Validation_Accuracy >= best_val_accuracy):
        torch.save(model.state_dict(), directory_path + 'models/fake_tweet2.ckpt')
        best_val_accuracy = Validation_Accuracy
        print('Model Saved')
