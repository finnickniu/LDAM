import os
from typing import Tuple
from numpy.core.fromnumeric import shape
from sklearn.utils.extmath import weighted_mode
# os.chdir("/home/comp/cssniu/RAIM/models/")
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
from text_lab.lab_text_dataloader import TEXTDataset
from torchtext import data 
from text_lab.fusion_cls import fusion_layer
from text_lab.text_bert import LEAM
from lab_testing.dataloader import knowledge_dataloader 

import numpy as np
import torch.nn.utils.rnn as rnn_utils

from sklearn import metrics
import warnings
import copy
import torch.nn.functional as F
import math
from transformers import AdamW
os.environ['CUDA_VISIBLE_DEVICES']="0,2"
warnings.filterwarnings('ignore')
torch.multiprocessing.set_sharing_strategy('file_system')

### GPU 22 cs flatten fixed, GPU 22 cs flatten not fixed, GPU 22 ca flatten fixed GPU 24 ca flatten not fixed,
num_epochs = 15
BATCH_SIZE = 3
Test_batch_size = 6
save_dir= "weights_fusion"
Flatten = True
Fixed  = True
strict = True
pretrained = False
save_name = "fusion_915_fixed"
weight_dir = ".pth"
device1 = "cuda:1" if torch.cuda.is_available() else "cpu"
device1 = torch.device(device1)
device2 = "cuda:0" if torch.cuda.is_available() else "cpu"
device2 = torch.device(device2)
Best_loss = 100
Bess_acc = 0
start_epoch = 0
hyperparams = {
               'num_epochs':num_epochs,
               'embedding_dim' : 768,
               'fusion_dim':300,
               "output_dim":25,
               'ngram':3,
               'dropout' : 0.5,
               'batch_size' : BATCH_SIZE,
               'device1':device1,
               'device2':device2}

def calc_loss_c(c,criterion,model, y, device):
    """
    torch.tensor([0,1,2]) is decoded identity label vector
    """
    f2_c = model.text_fc(c)
    # f2_c = model.fc(c)
    y_c =  torch.stack([torch.range(0, y.shape[1] - 1, dtype=torch.long)]*c.shape[0]).to(device)
    return criterion(f2_c,y_c)


def fit(epoch,model,train_iterator,test_iterator,fixed_label_embedding,fixed_task_embedding,optimizer,criterion,criterion1,hyperparams,flag = "train"):
    global Best_loss,Bess_acc,Fixed,Flatten,save_name,save_dir


    if flag == "train":
        device = hyperparams['device1']
        model.train()
        data_iter = train_iterator
    else:
        device = hyperparams['device2']
        model.eval()
        data_iter = test_iterator

    fixed_label_embedding = fixed_label_embedding.to(device).transpose(1,0)
    fixed_task_embedding = fixed_task_embedding.to(device).transpose(1,0)

    model.to(device)

    criterion.to(device)
    criterion1.to(device)

    loss_ls = []
    acc_ls = []
    f1_ls=[]


    for i,(data,length,label,text,task_token,label_token) in enumerate(tqdm(data_iter,desc=f"{flag}ing model")):
        optimizer.zero_grad()

        text_x = [t.to(device) for t in text]
        label_token = [l.to(device) for l in label_token]
        task_token = [t.to(device) for t in task_token]
        lab_x = data.to(device,dtype=torch.float)
        y= label.to(device,dtype=torch.float).squeeze()
        fixed_label_embedding_batch = fixed_label_embedding.repeat(lab_x.shape[0],1,1)
        fixed_task_embedding_batch = fixed_task_embedding.repeat(lab_x.shape[0],1,1)
        if flag == "train":
            
            with torch.set_grad_enabled(True):
                pred,c,t,text_pred,weights,fused_score,weighted_embed,c_o,g,u1  = model(text_x,label_token,task_token,lab_x,length,fixed_label_embedding_batch,fixed_task_embedding_batch,Fixed,Flatten,mode='fusion')
               
                fused_score = fused_score[0:1,:].squeeze(0).tolist()
                # print(fused_score)

                loss_v = criterion1(pred, y)

                loss_c = calc_loss_c(c,criterion,model,y,device)
                # print(loss_v.data)
                loss = loss_v + loss_c
                # print(f"loss classification: {float(loss_v.cpu().data)} loss label: {float(loss_c.cpu().data)}")
                loss.backward(retain_graph=True)
                optimizer.step()

        else:
            with torch.no_grad():
                pred,c,t,text_pred,weights,fused_score,weighted_embed,co,g,u1  = model(text_x,label_token,task_token,lab_x,length,fixed_label_embedding_batch,fixed_task_embedding_batch,Fixed,Flatten,mode='fusion')


                loss_v = criterion1(pred, y)
                loss_c = calc_loss_c(c,criterion,model,y,device)
                loss = loss_v + loss_c
                # loss = loss_v

        y = np.array(y.tolist())
        pred = np.array(pred.tolist())
        try:
            pred=(pred > 0.5) 
            f1 = metrics.f1_score(y,pred,average="micro")

            acc = metrics.roc_auc_score(y,pred,average="micro")
            # print(f'loss :{float(loss.cpu().data)} acc: {acc}')
            acc_ls.append(acc)
            f1_ls.append(f1)

        except:
            pass
        loss_ls.append(float(loss.cpu().data))

    if flag == "test":
        PATH=f"logs/{save_dir}/{save_name}_epoch_{epoch}_loss_{round(np.mean(loss_ls),4)}_f1_{round(np.mean(acc_ls),4)}.pth"
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts, PATH)
    print("PHASE：{} EPOCH : {} | F1 : {} | ROC ： {} | LOSS : {}".format(flag,epoch + 1,  np.mean(f1_ls),np.mean(acc_ls), np.mean(loss_ls)))
    return model

  

def engine(hyperparams,model, train_iterator, test_iterator,fixed_label_embedding,fixed_task_embedding,optimizer,criterion,criterion1):
# def engine(scheduler,model, train_iterator, test_iterator,optimizer,criterion,criterion1):
    global start_epoch
    for epoch in range(start_epoch,hyperparams['num_epochs']):
        model = fit(epoch,model,train_iterator,test_iterator,fixed_label_embedding,fixed_task_embedding,optimizer,criterion,criterion1,hyperparams,flag = "train")
        # try:
        model = fit(epoch,model,train_iterator,test_iterator,fixed_label_embedding,fixed_task_embedding,optimizer,criterion,criterion1,hyperparams,flag = "test")
        # except:pass
        # scheduler.step()


def collate_fn(data):

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
   


    task_embedding,label_embedding= knowledge_dataloader.load_embeddings("")
    fixed_label_embedding = torch.stack(label_embedding)
    fixed_task_embedding = torch.stack(task_embedding)

    train_data =  TEXTDataset('',flag="train",all_feature=True)

    test_data = TEXTDataset('',flag="test",all_feature=True)

    print('len of train data:', len(train_data))             
    print('len of test data:', len(test_data)) 
    trainloader = torch.utils.data.DataLoader(train_data, drop_last=True,batch_size=hyperparams["batch_size"], shuffle =True,collate_fn=collate_fn, num_workers=12)
    testloader = torch.utils.data.DataLoader(test_data,drop_last=True, batch_size=Test_batch_size, shuffle =True,collate_fn=collate_fn, num_workers=12)
    model = fusion_layer(hyperparams["embedding_dim"],hyperparams['fusion_dim'],hyperparams["dropout"],hyperparams["ngram"])
    if pretrained:
        model.load_state_dict(torch.load(weight_dir,map_location=torch.device(device2)), strict=strict)

    optimizer = optim.Adam(model.parameters(True), lr = 1e-5)
    criterion = nn.CrossEntropyLoss()
    criterion1 = nn.BCELoss()

    engine(hyperparams,model,trainloader,testloader,fixed_label_embedding,fixed_task_embedding, optimizer,criterion,criterion1)
