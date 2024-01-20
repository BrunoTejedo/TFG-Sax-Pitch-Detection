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

for folder in os.listdir('./RESULTS'):
    print('Analysing '+folder+' results ...')
    path = './RESULTS/'+folder
    os.mkdir(path+'/metrics')
    level = folder[0:6]
    
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

        

