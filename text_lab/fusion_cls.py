import torch
import torch.nn as nn
import torch.nn.functional as F
from text_lab.text_bert import LEAM
from text_lab.channelwise_lstm import cw_lstm_model


class fusion_layer(nn.Module):
    def __init__(self,embedding_dim,fusion_dim,dropout,ngram,output_dim = 25):
        super(fusion_layer, self).__init__()
        self.lab_encoder = cw_lstm_model(ngram,fusion_dim)
        self.text_encoder = LEAM(fusion_dim,embedding_dim, output_dim, dropout, ngram)
        self.feature_number = 2
        self.class_number = output_dim
        self.drop_out = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(2*fusion_dim,fusion_dim),
            nn.Dropout(dropout))
        self.norm2 =nn.LayerNorm(64)
        self.dense = nn.Sequential(
        (nn.Linear(256,128,bias=True)),
        (nn.LeakyReLU()),
        (nn.Linear(128,64,bias=True)),
        (nn.LeakyReLU()))
        self.clss = self.classification_layer(1088,25)
        self.text_fc = nn.Sequential(
            nn.Linear(fusion_dim, output_dim)
            )
        self.avgpooling =  nn.AvgPool2d(kernel_size=(2, 1))

    def classification_layer(self,in_channels,out_channels):
        clss = nn.Linear(in_channels,out_channels,bias=True)
        return clss


    def forward(self,text_x,label_token,task_token,lab_x,length,fixed_label_embedding,fixed_task_embedding,Fixed,Flatten,mode='fusion'):

        text_pred,weights,c,t,u,weighted_embed = self.text_encoder(text_x,label_token,task_token)
        if Fixed:
            lab_predict,fused_score,g,u1 = self.lab_encoder(lab_x,length,fixed_label_embedding,fixed_task_embedding)
        else:
            lab_predict,fused_score,g,u1 = self.lab_encoder(lab_x,length,c,t)


        f_x = torch.cat((lab_predict,text_pred),1)
        
        output =  self.sigmoid(self.drop_out(self.text_fc(self.avgpooling(f_x).squeeze(1))))
        c_o = c
        c = self.text_encoder.dropout(self.text_encoder.fc(c))
        return output,c,t,u,weights,fused_score,text_pred,c_o,g,u1





















