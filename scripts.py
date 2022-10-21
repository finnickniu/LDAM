
import numpy as np
import os
import pandas as pd

from tqdm import tqdm


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
        "Pleurisy; pneumothorax; pulmonary collapse",
        "Septicemia (except in labor)",
        "Shock"]
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
        "Pleurisy; pneumothorax; pulmonary collapse",
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
data_dir = os.path.join("")
text_dir = os.path.join("")
data_list = os.listdir(data_dir)
text_list = os.listdir(text_dir)
reulst = []
for t in tqdm(text_list):
    a = False
    c = False
    m = False
    text_df = pd.read_csv(os.path.join(text_dir,t))
    y = text_df[label_list].values
    y_index = np.argwhere(y[0] ==1 ).squeeze()
    if not y_index.any():continue
    if type(y_index) == int: y_index = [y_index]
    try:
        label = [label_list[i] for i in y_index]
        if t in data_list:
            for l in label:
                if l in accute:
                    a = True
                elif l in chronic:
                    c =True
                elif l in mixed:
                    m = True
            if a and not c and not m:
                reulst.append(t)
    except:continue
print(reulst)




# '10132_episode1_timeseries.csv', '10484_episode1_timeseries.csv', '10515_episode1_timeseries.csv', '10624_episode3_timeseries.csv', '10642_episode2_timeseries.csv', '11036_episode1_timeseries.csv', '11288_episode2_timeseries.csv', '11477_episode1_timeseries.csv', '11717_episode1_timeseries.csv', '1234_episode1_timeseries.csv', '12756_episode1_timeseries.csv', '12762_episode1_timeseries.csv', '13680_episode1_timeseries.csv', '14109_episode1_timeseries.csv', '14585_episode2_timeseries.csv', '14757_episode3_timeseries.csv', '14828_episode1_timeseries.csv', '15665_episode1_timeseries.csv', '16142_episode1_timeseries.csv', '16498_episode1_timeseries.csv', '16757_episode1_timeseries.csv', '17487_episode1_timeseries.csv', '17639_episode1_timeseries.csv', '17766_episode1_timeseries.csv', '18321_episode2_timeseries.csv', '18602_episode1_timeseries.csv', '18754_episode1_timeseries.csv', '19029_episode1_timeseries.csv', '19029_episode7_timeseries.csv', '1907_episode1_timeseries.csv', '19216_episode4_timeseries.csv', '19418_episode2_timeseries.csv', '19485_episode1_timeseries.csv', '19568_episode1_timeseries.csv', '19706_episode1_timeseries.csv', '20105_episode1_timeseries.csv', '20253_episode1_timeseries.csv', '20471_episode1_timeseries.csv', '20606_episode1_timeseries.csv', '20849_episode1_timeseries.csv', '21979_episode1_timeseries.csv', '2265_episode1_timeseries.csv', '2349_episode2_timeseries.csv', '23802_episode1_timeseries.csv', '24009_episode1_timeseries.csv', '24129_episode1_timeseries.csv', '24129_episode3_timeseries.csv', '24161_episode1_timeseries.csv', '2425_episode3_timeseries.csv', '24264_episode1_timeseries.csv', '24280_episode1_timeseries.csv', '24562_episode2_timeseries.csv', '24622_episode3_timeseries.csv', '24687_episode3_timeseries.csv', '24806_episode1_timeseries.csv', '25483_episode1_timeseries.csv', '25712_episode1_timeseries.csv', '26016_episode1_timeseries.csv', '26628_episode1_timeseries.csv', '26857_episode1_timeseries.csv', '27201_episode1_timeseries.csv', '27234_episode1_timeseries.csv', '27247_episode3_timeseries.csv', '27304_episode1_timeseries.csv', '27513_episode1_timeseries.csv', '27595_episode1_timeseries.csv', '27606_episode1_timeseries.csv', '27999_episode1_timeseries.csv', '28028_episode1_timeseries.csv', '28104_episode1_timeseries.csv', '285_episode1_timeseries.csv', '28681_episode2_timeseries.csv', '28744_episode1_timeseries.csv', '28832_episode1_timeseries.csv', '28934_episode1_timeseries.csv', '28945_episode1_timeseries.csv', '29011_episode1_timeseries.csv', '29123_episode1_timeseries.csv', '29359_episode1_timeseries.csv', '29493_episode1_timeseries.csv', '29552_episode1_timeseries.csv', '29552_episode2_timeseries.csv', '29817_episode1_timeseries.csv', '29838_episode1_timeseries.csv', '30107_episode1_timeseries.csv', '30110_episode1_timeseries.csv', '30149_episode1_timeseries.csv', '30177_episode1_timeseries.csv', '30184_episode1_timeseries.csv', '3019_episode1_timeseries.csv', '30226_episode1_timeseries.csv', '30228_episode1_timeseries.csv', '30403_episode1_timeseries.csv', '30677_episode1_timeseries.csv', '30714_episode2_timeseries.csv', '30740_episode1_timeseries.csv', '30941_episode1_timeseries.csv', '30997_episode1_timeseries.csv', '31044_episode1_timeseries.csv', '31142_episode1_timeseries.csv', '31366_episode1_timeseries.csv', '31389_episode1_timeseries.csv', '31658_episode1_timeseries.csv', '31798_episode1_timeseries.csv', '31851_episode1_timeseries.csv', '31964_episode1_timeseries.csv', '32103_episode1_timeseries.csv', '32222_episode1_timeseries.csv', '3977_episode3_timeseries.csv', '42132_episode1_timeseries.csv', '43043_episode1_timeseries.csv', '43052_episode1_timeseries.csv', '43128_episode2_timeseries.csv', '44589_episode1_timeseries.csv', '45489_episode1_timeseries.csv', '46116_episode1_timeseries.csv', '46427_episode1_timeseries.csv', '47429_episode1_timeseries.csv', '47582_episode1_timeseries.csv', '48220_episode1_timeseries.csv', '49092_episode1_timeseries.csv', '50476_episode1_timeseries.csv', '50879_episode1_timeseries.csv', '523_episode1_timeseries.csv', '53193_episode1_timeseries.csv', '5377_episode1_timeseries.csv', '54054_episode1_timeseries.csv', '5448_episode2_timeseries.csv', '548_episode1_timeseries.csv', '54922_episode1_timeseries.csv', '54922_episode2_timeseries.csv', '5518_episode1_timeseries.csv', '55966_episode2_timeseries.csv', '56097_episode1_timeseries.csv', '56502_episode3_timeseries.csv', '57251_episode1_timeseries.csv', '57277_episode1_timeseries.csv', '58391_episode1_timeseries.csv', '5952_episode1_timeseries.csv', '59720_episode1_timeseries.csv', '59864_episode4_timeseries.csv', '6001_episode1_timeseries.csv', '60139_episode1_timeseries.csv', '60548_episode1_timeseries.csv', '6091_episode1_timeseries.csv', '61413_episode1_timeseries.csv', '61852_episode1_timeseries.csv', '63047_episode1_timeseries.csv', '64296_episode2_timeseries.csv', '6445_episode1_timeseries.csv', '64893_episode1_timeseries.csv', '64992_episode1_timeseries.csv', '66706_episode2_timeseries.csv', '6703_episode1_timeseries.csv', '67117_episode1_timeseries.csv', '67150_episode2_timeseries.csv', '68268_episode1_timeseries.csv', '6833_episode1_timeseries.csv', '68542_episode1_timeseries.csv', '68884_episode1_timeseries.csv', '68944_episode2_timeseries.csv', '68962_episode1_timeseries.csv', '69169_episode1_timeseries.csv', '69574_episode1_timeseries.csv', '69649_episode1_timeseries.csv', '69797_episode1_timeseries.csv', '70104_episode1_timeseries.csv', '70463_episode1_timeseries.csv', '7101_episode1_timeseries.csv', '71571_episode1_timeseries.csv', '7175_episode1_timeseries.csv', '7180_episode1_timeseries.csv', '71924_episode1_timeseries.csv', '72778_episode1_timeseries.csv', '73488_episode1_timeseries.csv', '73713_episode2_timeseries.csv', '74409_episode1_timeseries.csv', '74673_episode1_timeseries.csv', '74798_episode1_timeseries.csv', '74821_episode1_timeseries.csv', '75663_episode1_timeseries.csv', '75782_episode1_timeseries.csv', '75996_episode1_timeseries.csv', '7672_episode1_timeseries.csv', '77352_episode1_timeseries.csv', '77665_episode1_timeseries.csv', '78149_episode1_timeseries.csv', '78324_episode1_timeseries.csv', '78442_episode1_timeseries.csv', '79280_episode1_timeseries.csv', '80473_episode1_timeseries.csv', '83210_episode1_timeseries.csv', '83527_episode1_timeseries.csv', '83826_episode1_timeseries.csv', '84445_episode1_timeseries.csv', '84454_episode1_timeseries.csv', '860_episode1_timeseries.csv', '87809_episode1_timeseries.csv', '87913_episode1_timeseries.csv', '88499_episode1_timeseries.csv', '90391_episode1_timeseries.csv', '9115_episode1_timeseries.csv', '91199_episode1_timeseries.csv', '91284_episode1_timeseries.csv', '91313_episode1_timeseries.csv', '91551_episode1_timeseries.csv', '9231_episode1_timeseries.csv', '92625_episode1_timeseries.csv', '9357_episode1_timeseries.csv', '93593_episode2_timeseries.csv', '93836_episode1_timeseries.csv', '94575_episode2_timeseries.csv', '94696_episode2_timeseries.csv', '94959_episode1_timeseries.csv', '95909_episode1_timeseries.csv', '96260_episode1_timeseries.csv', '97061_episode1_timeseries.csv', '97396_episode1_timeseries.csv', '97560_episode1_timeseries.csv', '97582_episode1_timeseries.csv', '97625_episode1_timeseries.csv', '9778_episode1_timeseries.csv', '97974_episode1_timeseries.csv', '98038_episode1_timeseries.csv', '98185_episode1_timeseries.csv', '98342_episode1_timeseries.csv', '98805_episode1_timeseries.csv', '9912_episode1_timeseries.csv', '99408_episode1_timeseries.csv', '10821_episode1_timeseries.csv', '13018_episode1_timeseries.csv', '16141_episode1_timeseries.csv', '16150_episode1_timeseries.csv', '1660_episode1_timeseries.csv', '16880_episode1_timeseries.csv', '16904_episode1_timeseries.csv', '17605_episode1_timeseries.csv', '19913_episode2_timeseries.csv', '21362_episode1_timeseries.csv', '22749_episode1_timeseries.csv', '22849_episode1_timeseries.csv', '22849_episode2_timeseries.csv', '26061_episode1_timeseries.csv', '27800_episode8_timeseries.csv', '28671_episode1_timeseries.csv', '29303_episode1_timeseries.csv', '31136_episode2_timeseries.csv', '40526_episode2_timeseries.csv', '41195_episode1_timeseries.csv', '48812_episode1_timeseries.csv', '48843_episode1_timeseries.csv', '49392_episode1_timeseries.csv', '5384_episode1_timeseries.csv', '53299_episode1_timeseries.csv', '53707_episode1_timeseries.csv', '5384_episode2_timeseries.csv', '54090_episode1_timeseries.csv', '58483_episode1_timeseries.csv', '6828_episode2_timeseries.csv', '6872_episode1_timeseries.csv', '76267_episode1_timeseries.csv', '79422_episode1_timeseries.csv', '91167_episode1_timeseries.csv', '9555_episode1_timeseries.csv', '9555_episode4_timeseries.csv', '96463_episode1_timeseries.csv', '9782_episode1_timeseries.csv', '9782_episode4_timeseries.csv', '99213_episode1_timeseries.csv']
