# mir_eval implementation
import os
import utils
import mir_eval.melody as mir
import numpy as np
import shutil
import matplotlib.pyplot as plt

for folder in os.listdir('./RESULTS'):
    path = './RESULTS/'+folder
    #os.rmdir(path+'/metrics')
    shutil.rmtree(path+'/metrics')

VFA_level1 = []
VFA_level2 = []
VFA_level3 = []
VFA_crepe = []
VFA_yin = []
VFA_yinFFT = []
VR_level1 = []
VR_level2 = []
VR_level3 = []
VR_crepe = []
VR_yin = []
VR_yinFFT = []
RPA_level1 = []
RPA_level2 = []
RPA_level3 = []
RPA_crepe = []
RPA_yin = []
RPA_yinFFT = []
RCA_level1 = []
RCA_level2 = []
RCA_level3 = []
RCA_crepe = []
RCA_yin = []
RCA_yinFFT = []
OA_level1 = []
OA_level2 = []
OA_level3 = []
OA_crepe = []
OA_yin = []
OA_yinFFT = []
for folder in os.listdir('./RESULTS'):
    print('Analysing '+folder+' results ...')
    path = './RESULTS/'+folder
    os.mkdir(path+'/metrics')
    level = folder[0:6]
    algor = folder[11:15]
    
    voicing_recall = []
    voicing_false_alarm = []
    raw_pitch_accuracy = []
    raw_chroma_accuracy = []
    overall_accuracy = []
    for file in os.listdir(path+'/values'):
        audioname = file[:-11]
        print('Analyzing '+audioname+' sample '+ ' ...')
        pitch_times = utils.extract_csv(path+'/times/'+audioname+'_times.csv')
        pitch_values = utils.extract_csv(path+'/values/'+audioname+'_values.csv')
        pitch_confidence = utils.extract_csv(path+'/confidence/'+audioname+'_confidence.csv')
        notes = utils.extract_csv('./Audio/'+level+'/CSV/'+audioname+'_midinotes.csv')
        onsets = utils.extract_csv('./Audio/'+level+'/CSV/'+audioname+'_midionsets.csv')
        durations = utils.extract_csv('./Audio/'+level+'/CSV/'+audioname+'_mididurations.csv')
        ref_times, ref_values = utils.convert_timestamps(pitch_times, pitch_values, notes, onsets, durations)
        
        # pitch_confidence values check
        for i in range(len(pitch_confidence)):
            if pitch_confidence[i]>1 or pitch_confidence[i]<0:
                pitch_confidence[i] = 0.0
                
        # to_cent_voicing
        ref_voicing, ref_cent, est_voicing, est_cent = mir.to_cent_voicing(ref_times, ref_values, pitch_times, pitch_values, est_voicing=pitch_confidence)
        
        # checks
        mir.validate_voicing(ref_voicing, est_voicing)
        mir.validate(ref_voicing, ref_cent, est_voicing, est_cent)
        
        # all metrics together
        evaluate = mir.evaluate(ref_times, ref_cent, pitch_times, est_cent, est_voicing=est_voicing)
        print(evaluate)
        voicing_recall.append(evaluate['Voicing Recall'])
        voicing_false_alarm.append(evaluate['Voicing False Alarm'])
        raw_pitch_accuracy.append(evaluate['Raw Pitch Accuracy'])
        raw_chroma_accuracy.append(evaluate['Raw Chroma Accuracy'])
        overall_accuracy.append(evaluate['Overall Accuracy'])
    
    
    voicing_recall = np.array(voicing_recall)
    voicing_false_alarm = np.array(voicing_false_alarm)
    raw_pitch_accuracy = np.array(raw_pitch_accuracy)
    raw_chroma_accuracy = np.array(raw_chroma_accuracy)
    overall_accuracy = np.array(overall_accuracy)
    
    # CSV FILES
    utils.save_metrics_csv(path+'/metrics/VR.csv', voicing_recall)
    utils.save_metrics_csv(path+'/metrics/VFA.csv', voicing_false_alarm)
    utils.save_metrics_csv(path+'/metrics/RPA.csv', raw_pitch_accuracy)
    utils.save_metrics_csv(path+'/metrics/RCA.csv', raw_chroma_accuracy)
    utils.save_metrics_csv(path+'/metrics/OA.csv', overall_accuracy)
    
    # Interesting stats
    print("Voicing Recall rate STATS:")
    print("----- max: ", np.max(voicing_recall))
    print("----- min: ", np.min(voicing_recall))
    print("----- mean: ", np.mean(voicing_recall))
    print("")
    print("Voicing False Alarm rate STATS:")
    print("----- max: ", np.max(voicing_false_alarm))
    print("----- min: ", np.min(voicing_false_alarm))
    print("----- mean: ", np.mean(voicing_false_alarm))
    print("")
    print("Raw Pitch Accuracy STATS:")
    print("----- max: ", np.max(raw_pitch_accuracy))
    print("----- min: ", np.min(raw_pitch_accuracy))
    print("----- mean: ", np.mean(raw_pitch_accuracy))
    print("")
    print("Raw Chroma Accuracy STATS:")
    print("----- max: ", np.max(raw_chroma_accuracy))
    print("----- min: ", np.min(raw_chroma_accuracy))
    print("----- mean: ", np.mean(raw_chroma_accuracy))
    print("")
    print("Overall Accuracy STATS:")
    print("----- max: ", np.max(overall_accuracy))
    print("----- min: ", np.min(overall_accuracy))
    print("----- mean: ", np.mean(overall_accuracy))
    print("")
    
    # Save stats
    if int(level[-1])==1:
        for i in range(len(overall_accuracy)):
            RCA_level1.append(raw_chroma_accuracy[i])
            RPA_level1.append(raw_pitch_accuracy[i])
            OA_level1.append(overall_accuracy[i])
            VFA_level1.append(voicing_false_alarm[i])
            VR_level1.append(voicing_recall[i])
    elif int(level[-1])==2:
        for i in range(len(overall_accuracy)):
            RCA_level2.append(raw_chroma_accuracy[i])
            RPA_level2.append(raw_pitch_accuracy[i])
            OA_level2.append(overall_accuracy[i])
            VFA_level2.append(voicing_false_alarm[i])
            VR_level2.append(voicing_recall[i])
    else:
        for i in range(len(overall_accuracy)):
            RCA_level3.append(raw_chroma_accuracy[i])
            RPA_level3.append(raw_pitch_accuracy[i])
            OA_level3.append(overall_accuracy[i])
            VFA_level3.append(voicing_false_alarm[i])
            VR_level3.append(voicing_recall[i])
    
    if algor=='CREP':
        for i in range(len(overall_accuracy)):
            RCA_crepe.append(raw_chroma_accuracy[i])
            RPA_crepe.append(raw_pitch_accuracy[i])
            OA_crepe.append(overall_accuracy[i])
            VFA_crepe.append(voicing_false_alarm[i])
            VR_crepe.append(voicing_recall[i])
    elif algor=='YinF':
        for i in range(len(overall_accuracy)):
            RCA_yinFFT.append(raw_chroma_accuracy[i])
            RPA_yinFFT.append(raw_pitch_accuracy[i])
            OA_yinFFT.append(overall_accuracy[i])
            VFA_yinFFT.append(voicing_false_alarm[i])
            VR_yinFFT.append(voicing_recall[i])
    else:
        for i in range(len(overall_accuracy)):
            RCA_yin.append(raw_chroma_accuracy[i])
            RPA_yin.append(raw_pitch_accuracy[i])
            OA_yin.append(overall_accuracy[i])
            VFA_yin.append(voicing_false_alarm[i])
            VR_yin.append(voicing_recall[i])
    
    # Plots
    array = np.arange(1,len(voicing_recall)+1)
    
    plt.scatter(array, voicing_recall, label='VR')
    plt.scatter(array, voicing_false_alarm, label='VFA')
    plt.scatter(array, raw_pitch_accuracy, label='RPA')
    plt.scatter(array, raw_chroma_accuracy, label='RCA')
    plt.scatter(array, overall_accuracy, label='OA')
    plt.legend()
    plt.savefig(path+'/metrics/metrics_results.png')
    plt.clf()

print("Saving metrics boxplots ...")
print(VR_yin)
print(VR_yinFFT)
print(VFA_yin)
print(VFA_yinFFT)
print(OA_yin)
print(OA_yinFFT)
print(RCA_yin)
print(RCA_yinFFT)
print(RPA_yin)
print(RPA_yinFFT)
print(VR_level1)
print(VR_level2)
print(VFA_level1)
print(VFA_level2)
print(OA_level1)
print(OA_level2)
print(RCA_level1)
print(RCA_level2)
print(RPA_level1)
print(RPA_level2)
shutil.rmtree('./boxplots')
os.mkdir('./boxplots')
###### level boxplots
# VR
fig = plt.figure()
fig.suptitle('VR', fontsize=14, fontweight='bold')
box_plot_data=[VR_level1,VR_level2,VR_level3]
plt.boxplot(box_plot_data, labels=['level1','level2','level3'])
plt.savefig('./boxplots/VR_levels.png')

# VFA
fig = plt.figure()
fig.suptitle('VFA', fontsize=14, fontweight='bold')
box_plot_data=[VFA_level1,VFA_level2,VFA_level3]
plt.boxplot(box_plot_data, labels=['level1','level2','level3'])
plt.savefig('./boxplots/VFA_levels.png')

# RPA
fig = plt.figure()
fig.suptitle('RPA', fontsize=14, fontweight='bold')
box_plot_data=[RPA_level1,RPA_level2,RPA_level3]
plt.boxplot(box_plot_data, labels=['level1','level2','level3'])
plt.savefig('./boxplots/RPA_levels.png')

# RCA
fig = plt.figure()
fig.suptitle('RCA', fontsize=14, fontweight='bold')
box_plot_data=[RCA_level1,RCA_level2,RCA_level3]
plt.boxplot(box_plot_data, labels=['level1','level2','level3'])
plt.savefig('./boxplots/RCA_levels.png')

# OA
fig = plt.figure()
fig.suptitle('OA', fontsize=14, fontweight='bold')
box_plot_data=[OA_level1,OA_level2,OA_level3]
plt.boxplot(box_plot_data, labels=['level1','level2','level3'])
plt.savefig('./boxplots/OA_levels.png')


###### algorithm boxplots
# VR
fig = plt.figure()
fig.suptitle('VR', fontsize=14, fontweight='bold')
box_plot_data=[VR_yin,VR_yinFFT,VR_crepe]
plt.boxplot(box_plot_data, labels=['yin','yinFFT','CREPE'])
plt.savefig('./boxplots/VR_algorithms.png')

# VFA
fig = plt.figure()
fig.suptitle('VFA', fontsize=14, fontweight='bold')
box_plot_data=[VFA_yin,VFA_yinFFT,VFA_crepe]
plt.boxplot(box_plot_data, labels=['yin','yinFFT','CREPE'])
plt.savefig('./boxplots/VFA_algorithms.png')

# RPA
fig = plt.figure()
fig.suptitle('RPA', fontsize=14, fontweight='bold')
box_plot_data=[RPA_yin,RPA_yinFFT,RPA_crepe]
plt.boxplot(box_plot_data, labels=['yin','yinFFT','CREPE'])
plt.savefig('./boxplots/RPA_algorithms.png')

# RCA
fig = plt.figure()
fig.suptitle('RCA', fontsize=14, fontweight='bold')
box_plot_data=[RCA_yin,RCA_yinFFT,RCA_crepe]
plt.boxplot(box_plot_data, labels=['yin','yinFFT','CREPE'])
plt.savefig('./boxplots/RCA_algorithms.png')

# OA
fig = plt.figure()
fig.suptitle('OA', fontsize=14, fontweight='bold')
box_plot_data=[OA_yin,OA_yinFFT,OA_crepe]
plt.boxplot(box_plot_data, labels=['yin','yinFFT','CREPE'])
plt.savefig('./boxplots/OA_algorithms.png')        

