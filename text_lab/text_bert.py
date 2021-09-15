
from re import X
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import repeat
import torch.nn.utils.rnn as rnn_utils
import numpy as np
# from lab_text_dataloader import TEXTDataset
from transformers import AutoTokenizer, AutoModel
import random
from tqdm import tqdm
import math
from text_lab.cat_embedding import cat_bert_embedding
# random.seed(2020)

class LabelWordCompatLayer(nn.Module):
    def __init__(self,fc,embedding_dim,ngram, output_dim):
        nn.Module.__init__(self)
        self.encoder = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim

        assert ngram % 2 == 1, "n-gram should be odd number {2r+1}"

        self.phrase_filter = nn.Conv2d(
            # dilation= 2,
            in_channels=1,
            out_channels=1,
            padding='same',
            kernel_size=(ngram,1))
        self.phrase_extract = nn.MaxPool2d(kernel_size=(1, output_dim))
        self.mp = nn.MaxPool1d(kernel_size=10)
        self.dropout = nn.Dropout(0.3)
        self.fc = fc
        self.fc1 = nn.Linear(embedding_dim,embedding_dim)

    def scaled_attention(self,v,c):
        v = self.fc(v)
        c = self.fc(c)
        B, Nt, E = v.shape
        v = v / math.sqrt(E)

        g = torch.bmm(v, c.transpose(-2, -1))
        return g

    def embedding(self,x,flag = "text"):
        batch_output = []
        embedded = []
        if flag == "text":
            for b in x:
                output = cat_bert_embedding(self.fc1,self.encoder,b)
                batch_output.append(output)
            max_length = max([i.shape[1] for i in batch_output])
            padding_batch = []
            for t in batch_output:
                if t.shape[1] < max_length:
                    padding = torch.zeros((1,max_length-t.shape[1],768)).to(f"cuda:{t.get_device()}")
                    t = torch.cat([t,padding],dim=1)
                padding_batch.append(t)
            embedded = torch.cat(padding_batch,dim=0)
        else:
            inputs = x[0]
            output = self.encoder(**inputs).pooler_output
            embedded = output.repeat(len(x),1,1)
        return embedded

###########################################################

    def forward(self, text,c0,task_token):
        # c0 = c0[:text.shape[0],:].long() ## random c
        v = self.dropout(self.embedding(text,flag = "text"))
        # print(v.shape)
        c = self.dropout(self.embedding(c0,flag = "label"))
        t = self.dropout(self.embedding(task_token,flag = "task"))
        # batch, text tokens,25
        g = self.scaled_attention(v,c)
        u = torch.relu(self.phrase_filter(g.unsqueeze(1)).squeeze(1))  # [b, l, k]

        m = self.dropout(self.phrase_extract(u))  # [b, l, 1]
        b = torch.softmax(m, dim=1)  # [b, l, 1]

        return b,v,c,t,u


class LEAM(nn.Module):
    def __init__(self, fusion_dim,embedding_dim, output_dim, dropout, ngram):
        nn.Module.__init__(self)


        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, fusion_dim)
            )

        self.compat_model = LabelWordCompatLayer(
            fc = self.fc,
            embedding_dim = embedding_dim,
            ngram=ngram,
            output_dim=output_dim,
        )
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(embedding_dim)
        self.sigmoid = nn.Sigmoid()


    def forward(self, text,label_token,task_token):

        weight, embed,c,t,u = self.compat_model(text,label_token,task_token)
        weighted_embed = self.dropout(weight*embed) 
        weighted_embed_sumed = weighted_embed.sum(1)
        z = self.dropout(self.fc(weighted_embed_sumed)).unsqueeze(1)
        # z = self.sigmoid(z)
        return z,weight,c,t,u,weighted_embed

def collate_fn(data):
    """
    定义 dataloader 的返回值
    :param data: 第0维：data，第1维：label
    :return: 序列化的data、记录实际长度的序列、以及label列表
    """
    data.sort(key=lambda x: len(x[0]), reverse=True)
 
    data_length = [sq[0].shape[0] for sq in data]

    input_x = [i[0].tolist() for i in data]
    y = [i[1] for i in data]
    text = [i[2] for i in data]
    task_token =  [i[3] for i in data]
    label_token =  [i[4] for i in data]
    data = rnn_utils.pad_sequence([torch.from_numpy(np.array(x)) for x in input_x],batch_first = True, padding_value=0)
    return data.unsqueeze(-1), data_length, torch.tensor(y, dtype=torch.float32),text,task_token,label_token
if __name__ == "__main__":
    device = torch.device("cuda:3")
    dataset = TEXTDataset('/home/comp/cssniu/RAIM/benchmark_data/all/data/train/',flag="train",all_feature=True)
    batch_size = 10
    embedding_dim = 768 
    output_dim = 25
    dropout = 0.5
    ngram = 3
    model = LEAM(embedding_dim, output_dim, dropout, ngram, batch_size).to(device)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    for (data,length,label,text,task_token,label_token) in tqdm(trainloader):
        text = [t.to(device) for t in text]
        label_token = [l.to(device) for l in label_token]
        z,weight,c,t = model(text,label_token)
        # print(pred)
        # print(pred)
    # print(output[0].shape)
















