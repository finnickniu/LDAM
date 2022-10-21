import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

from itertools import repeat
import numpy as np
import math



class cw_lstm_model(nn.Module):
    def __init__(self,ngram,fusion_dim):
        super(cw_lstm_model, self).__init__()
        self.drop_out = nn.Dropout(0.1)
        self.lstm1 =  nn.GRU(input_size=1, batch_first=True,hidden_size=fusion_dim, num_layers=1, bidirectional=True)
        self.lstm2 =  nn.GRU(input_size=fusion_dim, batch_first=True,hidden_size=fusion_dim, num_layers=1, bidirectional=True)
        self.norm1 = nn.LayerNorm(fusion_dim)
        self.flatten = nn.Flatten()
        self.phrase_filter = nn.Conv2d(
                                        in_channels=1,
                                        out_channels=1,
                                        padding='same',
                                        kernel_size=(ngram,1))
        self.phrase_extract = nn.MaxPool2d(kernel_size=(1, 17))
        self.phrase_extract = nn.MaxPool2d(kernel_size=(1, 25))
        self.relu = nn.LeakyReLU()

        self.clss = self.classification(1088,25)
        self.drop_out1 = nn.Dropout(0.1)

        self.dense = nn.Sequential(
        (nn.Linear(256,128,bias=True)),
        (nn.LeakyReLU()),
        (nn.Linear(128,64,bias=True)),
        (nn.LeakyReLU()))
        self.map_dense =  nn.Linear(2,1,bias=False)  
        self.norm2 =nn.LayerNorm(64)
        self.softmax = nn.Softmax(dim=1)


    
    def classification(self,in_channels,out_channels):
        clss = nn.Linear(in_channels,out_channels,bias=True)
        return clss
    # x = F.max_pool1d(x.float().unsqueeze(1),kernel_size = int(x.shape[-1]/hyperparams["token_length"])).squeeze(1).long()

    def self_att(self,q,k):

        B, Nt, E = q.shape
        q = q / math.sqrt(E)
        # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
        g = torch.bmm(q, k.transpose(-2, -1))
        u = torch.relu(self.phrase_filter(g.unsqueeze(1)).squeeze())  # [b, l, k]
        m = self.drop_out(F.max_pool2d(u,kernel_size = (1,u.shape[-1])))  # [b, l, 1]
        # m = self.drop_out(self.phrase_extract(u))  # [b, l, 1]
        b = torch.softmax(m, dim=1)  # [b, l, 1
        return b

    def cross_att(self,q,k):

        B, Nt, E = q.shape
        q = q / math.sqrt(E)
        # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
        g = torch.bmm(q, k.transpose(-2, -1))

        u = self.drop_out(self.relu(self.phrase_filter(g.unsqueeze(1)).squeeze(0)))  # [b, l, k]

        m = self.drop_out(F.max_pool2d(u,kernel_size = (1,u.shape[-1]))).squeeze(1)  # [b, l, 1]
        b = torch.softmax(m, dim=1)  # [b, l, 1]
        # print(b[0:1,:].squeeze(0).tolist())
        return b,g,u

    
### transpose permute 是维度交换, view 可以用于元素压缩和展开
    def forward(self,x,length,label_embedding,task_embedding):
    # def forward(self,x,length,score_ca):

        total_output = []
        for i in range(x.shape[2]): 
            input_x = x[:,:,i:i+1,:]

            input_x = input_x.squeeze(-1).float()
            # input_x = F.normalize(input_x, p=2, dim=2)
            pack= rnn_utils.pack_padded_sequence(input_x, torch.FloatTensor(length),batch_first=True)
            output, hidden = self.lstm1(pack)
            padded_output, others  = rnn_utils.pad_packed_sequence(output, batch_first=True)
            padded_output = padded_output.view(padded_output.shape[0],padded_output.shape[1],2, padded_output.shape[-1]//2)
            padded_output = torch.sum(padded_output,axis = 2)
            # last_hidden = padded_output[:,-1:,0:1,:] + padded_output[:,:1,1:2,:]
            # last_hidden = last_hidden.squeeze(1).squeeze(1)
            total_output.append(padded_output)
        total_output = torch.stack(total_output)
        total_output = torch.transpose(total_output,1,0).contiguous()
        total_output_shaped = total_output.view(total_output.shape[0]*total_output.shape[1],total_output.shape[2], padded_output.shape[-1])
        
        total_output_shaped,hn = self.lstm2(total_output_shaped)
        last_hidden = torch.sum(hn,axis=0).squeeze(0).contiguous()
        last_hidden = last_hidden.view(total_output.shape[0],total_output.shape[1],last_hidden.shape[-1])


        fused_score,g,u = self.cross_att(task_embedding,label_embedding)
        last_hidden_att = last_hidden * fused_score.squeeze(1)

        last_hidden_att =  self.norm1(self.drop_out1(last_hidden_att) + last_hidden)
        last_hidden_att = last_hidden_att.sum(1).unsqueeze(1)
        return last_hidden_att,fused_score,g,u
      














