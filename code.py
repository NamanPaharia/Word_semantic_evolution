"""
Original file is located at
    https://colab.research.google.com/drive/1wuquOL3njAK5OkCGi5T_Mi2Gim-WVFv0
"""

#pip install the requirments first


# Commented out IPython magic to ensure Python compatibility.
# %%capture
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm 
import re
import string
import torch
import random
import os
import nltk
from nltk.corpus import stopwords
from nltk import sent_tokenize
from collections import Counter
#pip install transformers
#pip install pytorch-pretrained-bert
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
import transformers
from transformers import *
#!wandb login


#loading scibert model
tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
model.eval()


# loading dataset
infile = open("/enc/uprod_mMDNMb/work/Copy",'rb')
data = pickle.load(infile)
infile.close()


#making batches of time slices
batch1 = data[data["Year"]<=1995]
#batch2 = data[operator.and_(data.Year>1995, data.Year<=2000)]
#batch3 = data[operator.and_(data.Year>2000, data.Year<=2005)]
#batch4 = data[operator.and_(data.Year>2005, data.Year<=2010)]
#batch5 = data[operator.and_(data.Year>2010, data.Year<=2015)]



#data transformation  to desired format
def lower_case(input_str):
  input_str = input_str.lower()
  return input_str

def change_to_bert_format(df):
  df['Body_Text'] = df['Body_Text'].apply(lambda x: lower_case(x))
  df['Body_Text'] = df['Body_Text'].apply(lambda x: re.sub('[^a-zA-z0-9\s.]','',x))

  text2 = df.drop(["Lemmatized_Body_Tokens_List","Title"],axis=1)
  text = pd.DataFrame(text2, columns = ["Year","Body_Text"])
  text = text.reset_index()
  text = text.drop(["index"],axis=1)

  text_dict = text.to_dict()
  len_text = len(text_dict["Year"])

  Year_list  = []
  Body_Text_list = []
  for i in tqdm(range(0,len_text)):
      Year = text_dict["Year"][i]
      Body_Text = text_dict["Body_Text"][i].split('.')
      for b in Body_Text:
          Year_list.append(Year)
          Body_Text_list.append(b)

  df_sentences = pd.DataFrame({"Year":Year_list},index=Body_Text_list)
  df_sentences.head()

  df_sentences = df_sentences["Year"].to_dict()
  df_sentences_list = list(df_sentences.keys())
  df_sentences_list = [str(d) for d in tqdm(df_sentences_list)]

  return df_sentences_list



#extracting words with freq> 100 and filtering all except the nouns from entire corpus
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
nouns = {x.name().split('.', 1)[0] for x in wn.all_synsets('n')}

def extract_vocabulary(sentences):
    stop_words = set(stopwords.words('english'))
    stop_words.update(['also'])
    punctuation = set(string.punctuation)
    vocab=[]

    for index, sentence in enumerate(sentences):
        if index%1000==0:
            print(str(index) + '/' + str(len(sentences)))
        words = re.sub("[^\w]", " ", sentence).split()
        for word in words:
            word_clean = ''
            for elem in word:
                if elem not in punctuation and not elem.isdigit():
                    word_clean += elem
            if len(word_clean) > 1 and word_clean not in stop_words:  # delete all words with only one character
                vocab.append(word_clean)
    vocab = Counter(vocab)
    type(vocab)
    for x in list(vocab):
      if(vocab[x]<100):
        del vocab[x]
    for word in list(vocab):
      if((word in nouns)==False):
        del vocab[word]
        return vocab
# output - vocab

#getting_vocab
data_sen  = change_to_bert_format(data)
vocab = extract_vocabulary(data_sen)




#get embedding of a sentence
def get_embedding_for_sentence(tokenized_sent):
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_sent)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_ids = [1] * len(tokenized_sent)
    segments_tensors = torch.tensor([segments_ids])
    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensors)
        batch_i = 0
        token_embeddings = []
        # For each token in the sentence...
        for token_i in range(len(tokenized_sent)):
            hidden_layers = []
            # For each of the 12 layers...
            for layer_i in range(len(encoded_layers)):
                # Lookup the vector for `token_i` in `layer_i`
                vec = encoded_layers[layer_i][batch_i][token_i]
                hidden_layers.append(vec)
            token_embeddings.append(hidden_layers)
        #concatenated_last_4_layers = [torch.cat((layer[-1], layer[-2], layer[-3], layer[-4]), 0) for layer in token_embeddings]
        summed_last_4_layers = [torch.sum(torch.stack(layer)[-4:], 0) for layer in token_embeddings]
        last_layer = [layer[-1] for layer in token_embeddings]
        return summed_last_4_layers




#get embedding for a word in corpus
def get_embeddings_for_word(word, sentences):
    print("Getting BERT embeddings for word:", word)
    word_embeddings = []
    valid_sentences = []
    for i, sentence in enumerate(sentences):
            marked_sent = "[CLS] " + sentence + " [SEP]"
            tokenized_sent = tokenizer.tokenize(marked_sent)
            if word in tokenized_sent and len(tokenized_sent) < 512 and len(tokenized_sent) > 3:
                sent_embedding = get_embedding_for_sentence(tokenized_sent)
                word_indexes = list(np.where(np.array(tokenized_sent) == word)[0])
                for index in word_indexes:
                    word_embedding = np.array(sent_embedding[index])
                    word_embeddings.append(word_embedding)
                    valid_sentences.append(sentence)
    word_embeddings = np.array(word_embeddings)
    valid_sentences = np.array(valid_sentences)
    return word_embeddings, valid_sentences





#get embedding for all words and all its occurences in corpus
def get_embeddings_for_all_word(sentences):
  word_map_embeddings = {}
  word_map_sentence = {}

  for j, sentence in enumerate(sentences):
    marked_sent = "[CLS] " + sentence + " [SEP]"
    tokenized_sent = tokenizer.tokenize(marked_sent)
    if len(tokenized_sent) < 512 and len(tokenized_sent) > 3:
      sent_embedding = get_embedding_for_sentence(tokenized_sent)
      for i,word in enumerate(tokenized_sent):
        if vocab[word]>0 :
          #word_embedding  = [1,2,3]
          word_embedding = np.array(sent_embedding[i])
          if word in word_map_embeddings:
            word_map_embeddings[word] = np.vstack((word_map_embeddings[word], word_embedding))
            word_map_sentence[word] = np.vstack((word_map_sentence[word], sentence))
          else:
            word_map_embeddings[word] = word_embedding
            word_map_sentence[word] = sentence
  return word_map_embeddings,word_map_sentence

print("changing format")
#changing batch data to sentence list
batch1list = change_to_bert_format(batch1)
#batch2list = change_to_bert_format(batch2)
#batch3list = change_to_bert_format(batch3)
#batch4list = change_to_bert_format(batch4)
#batch5list = change_to_bert_format(batch5)

print("making dict")
#getting embedding map for all batch
batch1_emb , batch1_sen = get_embeddings_for_all_word(batch1list)
#batch2_emb , batch2_sen = get_embeddings_for_all_word(batch2list)
#batch3_emb , batch3_sen = get_embeddings_for_all_word(batch3list)
#batch4_emb , batch4_sen = get_embeddings_for_all_word(batch4list)
#batch5_emb , batch5_sen = get_embeddings_for_all_word(batch5list)




print("saving output")
#save output dict map as files
np.save('batch1_emb',batch1_emb)
np.save('batch2_emb',batch2_emb)
#np.save('batch3_emb',batch3_emb)
#np.save('batch4_emb',batch4_emb)
#np.save('batch5_emb',batch5_emb)
#np.save('batch1_sen',batch1_sen)
#np.save('batch2_sen',batch2_sen)
#np.save('batch3_sen',batch3_sen)
#np.save('batch4_sen',batch4_sen)
#np.save('batch5_sen',batch5_sen)



