import os
from numpy.core.fromnumeric import shape
# os.chdir("/home/comp/cssniu/RAIM/models/")
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
from text_lab.lab_text_dataloader import TEXTDataset
from torchtext import data 
from text_lab.fusion_cls import fusion_layer
from text_lab.text_bert import LEAM
from text_lab.channelwise_lstm import cw_lstm_model
from lab_testing.evaluation import all_metric
torch.multiprocessing.set_sharing_strategy('file_system')
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib
from lab_testing.dataloader import knowledge_dataloader 

import numpy as np
from transformers import AutoTokenizer, AutoModel

import torch.nn.utils.rnn as rnn_utils
import pandas as pd
from sklearn import metrics

from sklearn.manifold import TSNE
import warnings
import copy
import torch.nn.functional as F
import math
import seaborn as sns
import heapq
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
warnings.filterwarnings('ignore')

### GPU 23 avg pooling not fixed; GPU 22 avg pooling fixed, GPU 22 flatten fixed, GPU24 flatten not fixed
num_epochs = 1
BATCH_SIZE = 30
device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
Best_loss = 10000
Flatten = True
Fixed  = True
weights_dir = "logs/weights_fusion/ca_flatten_fixed_61_epoch_13_loss_0.2935_acc_0.8929.pth"
hyperparams = {
               'num_epochs':num_epochs,
               'embedding_dim' : 768,
               'fusion_dim':300,
               "output_dim":25,
               'ngram':3,
               'dropout' : 0.5,
               'batch_size' : BATCH_SIZE,
               'device1':device}
label_list = ["Acute and unspecified renal failure",
        "Acute cerebrovascular disease",
        "Acute myocardial infarction",
        "Complications of surgical procedures or medical care",
        "Fluid and electrolyte disorders",
        "Gastrointestinal hemorrhage",
        "Other lower respiratory disease",
        "Other upper respiratory disease",
        "Pleurisy; pneumothorax; pulmonary collapse",
        "Pneumonia (except that caused by tuberculosis or sexually transmitted disease)",
        "Respiratory failure; insufficiency; arrest (adult)",
        "Septicemia (except in labor)",
        "Shock",
        "Chronic kidney disease",
        "Chronic obstructive pulmonary disease and bronchiectasis",
        "Coronary atherosclerosis and other heart disease",
        "Diabetes mellitus without complication",
        "Disorders of lipid metabolism",
        "Essential hypertension",
        "Hypertension with complications and secondary hypertension",
        "Cardiac dysrhythmias",
        "Conduction disorders",
        "Congestive heart failure; nonhypertensive",
        "Diabetes mellitus with complications",
        "Other liver diseases",
        ]
all_feature_list = ['Capillary refill rate', 
        'Diastolic blood pressure',
       'Fraction inspired oxygen', 
       'Glascow coma scale eye opening',
       'Glascow coma scale motor response', 
       'Glascow coma scale total',
       'Glascow coma scale verbal response', 
       'Glucose', 
       'Heart Rate', 
       'Height',
       'Mean blood pressure', 
       'Oxygen saturation', 
       'Respiratory rate',
       'Systolic blood pressure', 
       'Temperature', 
       'Weight', 
       'pH']
tsne = TSNE(random_state=0, perplexity=10)
# sns.set(rc={'figure.figsize':(20,15)})
palette = sns.color_palette("bright",25)

palette1 = sns.color_palette("dark", 3)

def calc_loss_c(c,criterion,model, y, device):
    """
    torch.tensor([0,1,2]) is decoded identity label vector
    """
    ## 每个class 内部自己做cross entropy， 相当于做了25次， 也就是25个batch，python cross entropy 自带softmax,也不用做onehot
    # print(c.shape)
    f2_c = model.text_fc(c)
    # f2_c = model.fc(c)
    y_c =  torch.stack([torch.range(0, y.shape[1] - 1, dtype=torch.long)]*c.shape[0]).to(device)
    return criterion(f2_c,y_c)

def is_subtoken(word):
    if word[:2] == "##":
        return True
    else:
        return False

def detoeken(tokens):
    restored_text = []
    for i in range(len(tokens)):
        if not is_subtoken(tokens[i]) and (i+1)<len(tokens) and is_subtoken(tokens[i+1]):
            restored_text.append(tokens[i] + tokens[i+1][2:])
            if (i+2)<len(tokens) and is_subtoken(tokens[i+2]):
                restored_text[-1] = restored_text[-1] + tokens[i+2][2:]
        elif not is_subtoken(tokens[i]):
            restored_text.append(tokens[i])
    return restored_text

def decode_attetnion(restored_original_text,tokens,score_index):

    restored_text = []
    tokens = [tokens[i] for i in score_index]
    for i in range(len(tokens)):
        if not is_subtoken(tokens[i]) and (i+1)<len(tokens) and is_subtoken(tokens[i+1]):
            w = tokens[i] + tokens[i+1][2:]
            if w in restored_original_text:
                restored_text.append(w)

        if (i+2)<len(tokens) and is_subtoken(tokens[i+2]):
                w = restored_text[-1] + tokens[i+2][2:]
                if w in restored_original_text:
                    restored_text[-1] = w

        elif not is_subtoken(tokens[i]):
            restored_text.append(tokens[i])

    return restored_text


def fit(criterion,criterion1,model,case_study_data_list,fixed_label_embedding,fixed_task_embedding,hyperparams,flag = "test"):
    global Flatten,Fixed



    device = hyperparams['device1']
    model.eval()



    fixed_label_embedding = fixed_label_embedding.to(device).transpose(1,0)
    fixed_task_embedding = fixed_task_embedding.to(device).transpose(1,0)

    model.to(device)
    tsne_results = []
    label_result = []
    for case_study_data in case_study_data_list:
        data,length,y,text_x,task_token,label_token,string_token =  case_study_data[0],case_study_data[1],case_study_data[2],case_study_data[3],case_study_data[4],case_study_data[5],case_study_data[6]
        text_x = [text_x.to(device)]
        label_token = [label_token.to(device)]
        task_token = [task_token.to(device)]
        lab_x = data.to(device,dtype=torch.float).unsqueeze(0)
        y= y.to(device,dtype=torch.float).squeeze().unsqueeze(0)
        with torch.no_grad():
            pred,c,t,weights_text_all,weights,fused_score,weighted_embed,c_o = model(text_x,label_token,task_token,lab_x,length,fixed_label_embedding,fixed_task_embedding,Fixed,Flatten,mode='fusion')
    
            loss_v = criterion1(pred, y)
            loss_c = calc_loss_c(c,criterion,model,y,device)
            loss = loss_v + loss_c
            c,t,weights_text_all,weights,weighted_embed,fused_score,fixed_label_embedding,fixed_task_embedding = c_o.cpu().data,t.cpu().data,weights_text_all.cpu().data,weights.cpu().data,weighted_embed.cpu().data,fused_score.cpu().data,fixed_label_embedding.cpu().data,fixed_task_embedding.cpu().data
            y = np.array(y.tolist())
            pred = np.array(pred.tolist())
            pred=(pred > 0.5) 

            acc = metrics.f1_score(y,pred,average="micro")
            weights = weights[0:1,1:-1].squeeze(0).tolist()
            fused_score = fused_score.squeeze(0).tolist()
            max_num_index_fusion = list(map(fused_score.index, heapq.nlargest(len(fused_score), fused_score)))
            print(max_num_index_fusion)
            # weights = torch.softmax(weights_text_all[0,:,0:1],dim = 0).squeeze().tolist()


            max_num_index_weights = sorted(list(map(weights.index, heapq.nlargest(len(weights)//2, weights))))
            print([string_token[i] for i in max_num_index_weights])
            y_index = np.argwhere(y[0] ==1 ).squeeze()
            label = [label_list[i] for i in y_index]
            # print(label)
            # print(float(loss.cpu().data),acc)
            # X_tsne1 = tsne.fit_transform(weighted_embed.squeeze())
            # X_tsne2 = tsne.fit_transform(c.squeeze())
            # X_tsne3 = tsne.fit_transform(t.squeeze())
            # X_tsne1 = scaler.fit_transform(X_tsne1)
            # X_tsne2 = scaler.fit_transform(X_tsne2)
            # X_tsne3 = scaler.fit_transform(X_tsne3)
            # tsne_results.append(X_tsne1)
            # label_result.append(X_tsne2)

    # label_list1 = ["chronic_disease","acute_disease","mixed_disease"]
    # marker_list = ['x',"+","*"]
    # c = scaler.fit_transform(c.squeeze())
    t = scaler.fit_transform(t.squeeze())
    # X_tsne2 = tsne.fit_transform(c)
    X_tsne3 = tsne.fit_transform(t)

    # sns.scatterplot(X_tsne2[:,0].squeeze(), X_tsne2[:,1].squeeze())
    # for t in range(len(X_tsne2)):
    #     plt.text(X_tsne2[t:t+1,0].squeeze(),X_tsne2[t:t+1,1].squeeze()+0.05,range(25)[t])
    
    # sns.scatterplot(X_tsne3[:,0].squeeze(), X_tsne3[:,1].squeeze())
    for i in range(17):
        plt.scatter(X_tsne3[:, 0], X_tsne3[:, 1],color = "orange")
    for t in range(len(X_tsne3)):
        plt.text(X_tsne3[t:t+1,0].squeeze(),X_tsne3[t:t+1,1].squeeze()+0.05,range(17)[t],size = "large")
    plt.savefig(f'ns_models/images/tsne_task.jpg')

    # label_list1 = [0,1,2]

    # for i,t in enumerate(tsne_results): 

    #     sns.scatterplot(t[:,0].squeeze(), t[:,1].squeeze(),marker = marker_list[i],s=300,palette=palette1)
    # label_x = np.concatenate((label_result[0][3:6,0],label_result[1][15:16,0],label_result[1][17:19,0],label_result[1][20:21,0],label_result[2][22:23,0]),axis = 0)
    # label_y = np.concatenate((label_result[0][3:6,1],label_result[1][15:16,1],label_result[1][17:19,1],label_result[1][20:21,1],label_result[2][22:23,1]),axis = 0)
    # feature_list = [
    #    " Acute myocardial infarction", "Coronary atherosclerosis and other heart disease", "Disorders of lipid metabolism",
    #     "Essential hypertension", "Congestive heart failure; nonhypertensive", 'Cardiac dysrhythmias','Fluid and electrolyte disorders', 'Gastrointestinal hemorrhage'
    # ]

    # sns.scatterplot(label_x,label_y, hue=feature_list,s = 600,  palette=palette)
    # sns.scatterplot(label_result[1][:,0], label_result[1][:,1], hue=all_feature_list, s = 400,palette=palette)



        # X_tsne1 = tsne.fit_transform(c[1:2,:,:].squeeze())
            # X_tsne2 = tsne.fit_transform(t[1:2,:,:].squeeze())
            # X_tsne = np.concatenate((X_tsne1,X_tsne2),axis = 0)

            # sns.scatterplot(X_tsne1[:,0].squeeze(), X_tsne1[:,1].squeeze(), hue=label_list, s = 400,palette=palette)
            # sns.scatterplot(X_tsne2[:,0].squeeze(), X_tsne2[:,1].squeeze(), hue=all_feature_list,s = 300, marker = 'x', palette=palette1)

            # for i in range(X_tsne1.shape[0]):
            #     plt.scatter(X_tsne1[:, 0], X_tsne1[:, 1], color=plt.cm.Set1([i]))
            # for i in range(25,25+X_tsne2.shape[0]):
            #     plt.scatter(X_tsne2[:, 0], X_tsne2[:, 1], color=plt.cm.Set1([i]))
            # for i in range(X_tsne3.shape[0]):
            #     plt.scatter(X_tsne3[:, 0], X_tsne3[:, 1], color="yellow")
            # for i in range(X_tsne3.shape[0]):
            #     plt.scatter(X_tsne4[:, 0], X_tsne4[:, 1], color="blue")
            # print(X_tsne.shape)
            # label = np.unique(C)
            # ys = [i + i ** 2 for i in range(y.shape[1])]
            # colors = cm.rainbow(np.linspace(0, 1, len(ys)))

            # for y, c in zip(label, colors):
            #     print(y, c)
            #     x = reduced_mat[C == y]
            #     plt.scatter(x[:, 0], x[:, 1], color=red)
            # x_min, x_max = X_tsne.min(0), X_tsne.max(0)
            # X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
            # label = range(10000)
            # plt.figure(figsize=(8, 8))
            # for i in range(X_norm.shape[0]):
            #     plt.text(X_norm[i, 0], X_norm[i, 1], str(label[i]), color=plt.cm.Set1(label[i]), 
            # fontdict={'weight': 'bold', 'size': 9})
            # plt.savefig('ns_models/images/tsne_label.jpg')

            # break


  

# def engine(hyperparams,model,case_study_data,fixed_label_embedding,fixed_task_embedding,criterion,criterion1):
# def engine(scheduler,model, train_iterator, test_iterator,optimizer,criterion,criterion1):


        # scheduler.step()


def collate_fn(data):
    """
    定义 dataloader 的返回值
    :param data: 第0维：data，第1维：label
    :return: 序列化的data、记录实际长度的序列、以及label列表
    """
    data_length = [data[0].shape[0]]

    input_x = data[0].tolist()
    y =  data[1]
    text =  data[2]
    task_token =  data[3]
    label_token =   data[4]
    string_token =  data[5]
    data = rnn_utils.pad_sequence([torch.from_numpy(np.array(x)) for x in input_x],batch_first = True, padding_value=0)
    return data.unsqueeze(-1), data_length, torch.tensor(y, dtype=torch.float32),text,task_token,label_token,string_token


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT",do_lower_case=True)

    task_embedding,label_embedding= knowledge_dataloader.load_embeddings("/home/comp/cssniu/RAIM/embedding.pth")
    fixed_label_embedding = torch.stack(label_embedding)
    fixed_task_embedding = torch.stack(task_embedding)

    model = fusion_layer(hyperparams["embedding_dim"],hyperparams['fusion_dim'],hyperparams["dropout"],hyperparams["ngram"])
    model.load_state_dict(torch.load(weights_dir,map_location=torch.device(device)), strict=True)

    criterion = nn.CrossEntropyLoss()
    criterion1 = nn.BCELoss()
    best_f1 = 0
    best_f1_name =''
    # for test_name in tqdm(sorted(os.listdir('/home/comp/cssniu/RAIM/benchmark_data/all/data/test/'))):

    data_dir = os.path.join("/home/comp/cssniu/RAIM/benchmark_data/all/data/test")

    text_dir = os.path.join("/home/comp/cssniu/RAIM/benchmark_data/all/text/test")
    description_df =  pd.read_csv("/home/comp/cssniu/RAIM/benchmark_data/text/task_label_def.csv")
    test_name = '53193_episode1_timeseries.csv'
    # test_name = '9501_episode1_timeseries.csv'
    test_name_list = ["44_episode1_timeseries.csv","53193_episode1_timeseries.csv","10068_episode1_timeseries.csv"]
    test_name_list = ["44_episode1_timeseries.csv"]

    case_study_data_list = []
    for test_name in test_name_list:
        data = pd.read_csv(os.path.join(data_dir,test_name))[all_feature_list].values
        text_df = pd.read_csv(os.path.join(text_dir,test_name))
        text = text_df["TEXT_y"].values[0]
        # case_study_data = [data_file,lab_file]
        task = list(description_df["Description"].values[:-25])
        label = list(description_df["Description"].values[-25:])
        task_token = tokenizer(task, return_tensors="pt", padding=True)
        label_token = tokenizer(label, return_tensors="pt", padding=True)
        text_token = tokenizer(text, return_tensors="pt", padding=True)
        string_token = tokenizer.tokenize(text)
        y = text_df[label_list].values
        temp_data = [data,y,text_token,task_token,label_token,string_token]
        case_study_data_list.append(collate_fn(temp_data))
    fit(criterion,criterion1,model,case_study_data_list,fixed_label_embedding,fixed_task_embedding,hyperparams,flag = "test")
    # loss,acc = engine(hyperparams,model,case_study_data,fixed_label_embedding,fixed_task_embedding,criterion,criterion1)
        # if acc > best_f1:
        #     best_f1 = acc
        #     best_f1_name = test_name
    # print(best_f1_name,best_f1)