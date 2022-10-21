import os
from numpy.core.fromnumeric import shape
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
from text_lab.lab_text_dataloader import TEXTDataset
from torchtext import data 
from text_lab.fusion_cls import fusion_layer
from text_lab.channelwise_lstm import cw_lstm_model
from metrics import all_metric
torch.multiprocessing.set_sharing_strategy('file_system')
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib
from lab_testing.dataloader import knowledge_dataloader 
import pandas as pd
import numpy as np
import torch.nn.utils.rnn as rnn_utils

from sklearn.manifold import TSNE
import warnings
import copy
import torch.nn.functional as F
import math
import seaborn as sns
import heapq
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
num_epochs = 1
BATCH_SIZE = 30
device = "cuda:1" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
Best_loss = 10000
Flatten = True
Fixed  = True


weights_list = sorted(os.listdir("logs/weights_fusion/"))




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
all_feature_list = ['Capillary refill rate', 'Diastolic blood pressure',
       'Fraction inspired oxygen', 'Glascow coma scale eye opening',
       'Glascow coma scale motor response', 'Glascow coma scale total',
       'Glascow coma scale verbal response', 'Glucose', 'Heart Rate', 'Height',
       'Mean blood pressure', 'Oxygen saturation', 'Respiratory rate',
       'Systolic blood pressure', 'Temperature', 'Weight', 'pH']

accute = ["Acute and unspecified renal failure",
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
        "Shock",]
chronic =[
        "Chronic kidney disease",
        "Chronic obstructive pulmonary disease and bronchiectasis",
        "Coronary atherosclerosis and other heart disease",
        "Diabetes mellitus without complication",
        "Disorders of lipid metabolism",
        "Essential hypertension",
        "Hypertension with complications and secondary hypertension"]
mixed= [
        "Cardiac dysrhythmias",
        "Conduction disorders",
        "Congestive heart failure; nonhypertensive",
        "Diabetes mellitus with complications",
        "Other liver diseases",
        ]
tsne = TSNE(random_state=0, perplexity=10)
# sns.set(rc={'figure.figsize':(25,20)})
palette = sns.color_palette("bright", 25)

palette1 = sns.color_palette("dark", 25)

def calc_loss_c(c,criterion,model, y, device):
    """
    torch.tensor([0,1,2]) is decoded identity label vector
    """

    f2_c = model.text_fc(c)
    # f2_c = model.fc(c)
    y_c =  torch.stack([torch.range(0, y.shape[1] - 1, dtype=torch.long)]*c.shape[0]).to(device)
    return criterion(f2_c,y_c)

def is_subtoken(word):
    if word[:2] == "##":
        return True
    else:
        return False




def fit(w,pred_result,label_result,criterion,criterion1,epoch,model,testloader,fixed_label_embedding,fixed_task_embedding,hyperparams,flag = "test"):
    global Flatten,Fixed



    device = hyperparams['device1']
    model.eval()


    data_iter = testloader

    fixed_label_embedding = fixed_label_embedding.to(device).transpose(1,0)
    fixed_task_embedding = fixed_task_embedding.to(device).transpose(1,0)

    model.to(device)


    micro_auc_list = []
    macro_auc_list = []
    micro_f1_list = []
    macro_f1_list = []
    micro_precision_list = []
    macro_precision_list = []
    micro_recall_list = []
    macro_recall_list = []
    for i,(data,length,label,text,task_token,label_token,string_token) in enumerate(tqdm(data_iter,desc=f"{flag}ing model")):

        text_x = [t.to(device) for t in text]
        label_token = [l.to(device) for l in label_token]
        task_token = [t.to(device) for t in task_token]
        lab_x = data.to(device,dtype=torch.float)
        y= label.to(device,dtype=torch.float).squeeze()
        with torch.no_grad():
            pred,c,t,text_pred,weights,fused_score,weighted_embed,c_o,g,u1 = model(text_x,label_token,task_token,lab_x,length,fixed_label_embedding,fixed_task_embedding,Fixed,Flatten,mode='fusion')
            y = np.array(y.tolist())
            for l in range(y.shape[0]):
                ac = False
                ch = False
                mix = False
                y_index = np.argwhere(y[l,:] ==1 ).squeeze()
                if not y_index.any():continue
                if len(np.array([y_index]).shape) == 1: y_index = [y_index]

                label = [label_list[i] for i in y_index]  
                for la in label:
                    if la in accute:
                        ac = True
                    elif la in chronic:
                        ch =True
                    elif la in mixed:
                        mix = True
                append = False
                if ac and not ch and not mix:
                    label_result.append("acute")
                    append = True
                elif not ac and ch and not mix:
                    label_result.append("chronic")
                    append = True
                elif not ac and not ch and mix:
                    label_result.append("mixed")
                    append = True
                if append:
                    pred_result.append(np.array(weighted_embed.tolist())[l:l+1,].squeeze())
 
            pred = np.array(pred.tolist())
            c,t,text_pred,weights,g,u1 = c.cpu().data,t.cpu().data,text_pred.cpu().data,weights.cpu().data,g.squeeze().cpu().data,u1.squeeze().cpu().data
            # g = scaler.fit_transform(g)

            # sns.heatmap(g,cmap="Blues")
            # plt.savefig('ns_models/images/heat_map.jpg')
            # break

            
            weights = weights[0:1,:].squeeze(0).tolist()
            max_num_index_weights = list(map(weights.index, heapq.nlargest(150, weights)))
            fused_score = fused_score[0:1,:].squeeze(0).tolist()
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


            micro_auc,macro_auc,micro_f1,macro_f1,micro_precision,macro_precision,micro_recall,macro_recall = all_metric(y,pred)
            if micro_auc > 0:
                micro_auc_list.append(micro_auc)
            if macro_auc > 0:
                macro_auc_list.append(macro_auc)
            if micro_f1 > 0:
                micro_f1_list.append(micro_f1)
            if macro_f1 > 0:
                macro_f1_list.append(macro_f1)
            if micro_precision > 0:
                micro_precision_list.append(micro_precision)
            if macro_precision > 0:
                macro_precision_list.append(macro_precision)
            if micro_recall > 0:
                micro_recall_list.append(micro_recall)
            if macro_recall > 0:
                macro_recall_list.append(macro_recall)
    micro_auc_mean = np.array(micro_auc_list).mean()
    macro_auc_mean = np.array(macro_auc_list).mean()

    micro_f1_mean = np.array(micro_f1_list).mean()
    macro_f1_mean = np.array(macro_f1_list).mean()

    micro_precision_mean = np.array(micro_precision_list).mean()
    macro_precision_mean = np.array(macro_precision_list).mean()

    micro_recall_mean = np.array(micro_recall_list).mean()
    macro_recall_mean = np.array(macro_recall_list).mean()
    print('weights: ',w)
    print('micro roc auc: ',micro_auc_mean)
    print('macro roc auc: ',macro_auc_mean)
    print('micro f1: ',micro_f1_mean)
    print('macro f1: ',macro_f1_mean)
    print('micro precision: ',micro_precision_mean)
    print('macro precision: ',macro_precision_mean)
    print('micro recall: ',micro_recall_mean)
    print('macro recall: ',macro_recall_mean)

    print('-' * 20)
    f = 'note_912_fixed.txt'
    with open(f,"a") as file:

        file.write(f'weights : {w}'+"\n")
        file.write(f'micro roc auc : {micro_auc_mean}'+"\n")
        file.write(f'macro roc auc : {macro_auc_mean}'+"\n")
        file.write(f'micro f1 : {micro_f1_mean}'+"\n")
        file.write(f'macro f1 : {macro_f1_mean}'+"\n")
        file.write(f'micro precision : {micro_precision_mean}'+"\n")
        file.write(f'macro precision : {macro_precision_mean}'+"\n")
        file.write(f'micro recall : {micro_recall_mean}'+"\n")
        file.write(f'macro recall : {macro_recall_mean}'+"\n")
        file.write('-' * 20+"\n")





    return c

  

def engine(w,hyperparams,model,testloader,fixed_label_embedding,fixed_task_embedding,criterion,criterion1):
# def engine(scheduler,model, train_iterator, test_iterator,optimizer,criterion,criterion1):

    start_epoch = 0
    pred_result = []
    label_result = []
    for epoch in range(start_epoch,hyperparams['num_epochs']):
        c = fit(w,pred_result,label_result,criterion,criterion1,epoch,model,testloader,fixed_label_embedding,fixed_task_embedding,hyperparams,flag = "test")
        # scheduler.step()
    pred_result = np.array(pred_result)
    label_result = np.array(label_result)
    # X_tsne = TSNE(n_components=2,random_state=33).fit_transform(pred_result)
    # X_tsne = scaler.fit_transform(X_tsne)
    # pred_result = scaler.fit_transform(pred_result)

    # pred_df = pd.DataFrame(pred_result)
    # label_df = pd.DataFrame(label_result)
    # with open("data.tsv", 'w') as write_tsv:
    #     write_tsv.write(pred_df.to_csv(sep='\t', index=False,header=False))
    # with open("label.tsv", 'w') as write_tsv:
    #     write_tsv.write(label_df.to_csv(sep='\t', index=False,header=False))



    # sns.scatterplot(X_tsne[:,0].squeeze(), X_tsne[:,1].squeeze(), hue=label_result)

    # plt.savefig('ns_models/images/tsne_word_label.jpg')




def collate_fn(data):

    data.sort(key=lambda x: len(x[0]), reverse=True)
 
    data_length = [sq[0].shape[0] for sq in data]

    input_x = [i[0].tolist() for i in data]
    y = [i[1] for i in data]
    text = [i[2] for i in data]
    task_token =  [i[3] for i in data]
    label_token =  [i[4] for i in data]
    string_token = [i[5] for i in data]
    data = rnn_utils.pad_sequence([torch.from_numpy(np.array(x)) for x in input_x],batch_first = True, padding_value=0)
    return data.unsqueeze(-1), data_length, torch.tensor(y, dtype=torch.float32),text,task_token,label_token,string_token


if __name__ == "__main__":

    task_embedding,label_embedding= knowledge_dataloader.load_embeddings("")
    fixed_label_embedding = torch.stack(label_embedding)
    fixed_task_embedding = torch.stack(task_embedding)


    test_data = TEXTDataset('',flag="test",all_feature=True)

    print('len of test data:', len(test_data)) 
    testloader = torch.utils.data.DataLoader(test_data,drop_last=True, batch_size=BATCH_SIZE, shuffle =True,collate_fn=collate_fn, num_workers=12)

    model = fusion_layer(hyperparams["embedding_dim"],hyperparams['fusion_dim'],hyperparams["dropout"],hyperparams["ngram"])
    for w in tqdm(weights_list):
        weights_dir = os.path.join("logs/weights_fusion/",w)
        print(weights_dir)
        model.load_state_dict(torch.load(weights_dir,map_location=torch.device(device)), strict=True)
        criterion = nn.CrossEntropyLoss()
        criterion1 = nn.BCELoss()
        engine(w,hyperparams,model,testloader,fixed_label_embedding,fixed_task_embedding,criterion,criterion1)
