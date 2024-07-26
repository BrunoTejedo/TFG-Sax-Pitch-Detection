"""
mir_eval implementation

"""
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import mir_eval.melody as mir
import utils


def analyze_folder(path, folder, results, specific_results):
    level = folder[:6]
    algor = folder[11:15]
    
    if algor == 'CREP':
        algor = 'CREPE'
    elif algor == 'YinF':
        algor = 'YinFFT'
    else:
        algor = 'Yin'

    voicing_recall = []
    voicing_false_alarm = []
    raw_pitch_accuracy = []
    raw_chroma_accuracy = []
    overall_accuracy = []

    for file in os.listdir(os.path.join(path, 'values')):
        audioname = file[:-11]
        print(f'Analyzing {audioname} sample ...')

        pitch_times = utils.extract_csv(os.path.join(path, 'times', f'{audioname}_times.csv'))
        pitch_values = utils.extract_csv(os.path.join(path, 'values', f'{audioname}_values.csv'))
        pitch_confidence = utils.extract_csv(os.path.join(path, 'confidence', f'{audioname}_confidence.csv'))
        notes = utils.extract_csv(os.path.join('./Audio', level, 'CSV', f'{audioname}_midinotes.csv'))
        onsets = utils.extract_csv(os.path.join('./Audio', level, 'CSV', f'{audioname}_midionsets.csv'))
        durations = utils.extract_csv(os.path.join('./Audio', level, 'CSV', f'{audioname}_mididurations.csv'))
        ref_times, ref_values = utils.convert_timestamps(pitch_times, pitch_values, notes, onsets, durations)

        # pitch_confidence values check
        for i in range(len(pitch_confidence)):
            if pitch_confidence[i] > 1 or pitch_confidence[i] < 0:
                pitch_confidence[i] = 0.0

        # to_cent_voicing
        ref_voicing, ref_cent, est_voicing, est_cent = mir.to_cent_voicing(ref_times, ref_values, pitch_times,
                                                                           pitch_values, est_voicing=pitch_confidence)

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
    utils.save_metrics_csv(os.path.join(path, 'metrics', 'VR.csv'), voicing_recall)
    utils.save_metrics_csv(os.path.join(path, 'metrics', 'VFA.csv'), voicing_false_alarm)
    utils.save_metrics_csv(os.path.join(path, 'metrics', 'RPA.csv'), raw_pitch_accuracy)
    utils.save_metrics_csv(os.path.join(path, 'metrics', 'RCA.csv'), raw_chroma_accuracy)
    utils.save_metrics_csv(os.path.join(path, 'metrics', 'OA.csv'), overall_accuracy)

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

    results['VR'][level].extend(voicing_recall)
    results['VFA'][level].extend(voicing_false_alarm)
    results['RPA'][level].extend(raw_pitch_accuracy)
    results['RCA'][level].extend(raw_chroma_accuracy)
    results['OA'][level].extend(overall_accuracy)

    results['VR'][algor].extend(voicing_recall)
    results['VFA'][algor].extend(voicing_false_alarm)
    results['RPA'][algor].extend(raw_pitch_accuracy)
    results['RCA'][algor].extend(raw_chroma_accuracy)
    results['OA'][algor].extend(overall_accuracy)
    
    specific_results['VR'][level][algor].extend(voicing_recall)
    specific_results['VFA'][level][algor].extend(voicing_false_alarm)
    specific_results['RPA'][level][algor].extend(raw_pitch_accuracy)
    specific_results['RCA'][level][algor].extend(raw_chroma_accuracy)
    specific_results['OA'][level][algor].extend(overall_accuracy)

    # Plots
    array = np.arange(1, len(voicing_recall) + 1)

    plt.scatter(array, voicing_recall, label='VR')
    plt.scatter(array, voicing_false_alarm, label='VFA')
    plt.scatter(array, raw_pitch_accuracy, label='RPA')
    plt.scatter(array, raw_chroma_accuracy, label='RCA')
    plt.scatter(array, overall_accuracy, label='OA')
    plt.xlabel("Audio samples")
    plt.ylabel("Metric score")
    plt.legend()
    plt.savefig(os.path.join(path, 'metrics', 'metrics_results.png'))
    plt.clf()


def save_boxplots(results):
    shutil.rmtree('./boxplots', ignore_errors=True)
    os.mkdir('./boxplots')

    levels = ['LEVEL1', 'LEVEL2', 'LEVEL3']
    algorithms = ['Yin', 'YinFFT', 'CREPE']

    # Level boxplots
    for metric in ['VR', 'VFA', 'RPA', 'RCA', 'OA']:
        fig = plt.figure()
        fig.suptitle(metric, fontsize=14, fontweight='bold')
        box_plot_data = [results[metric][level] for level in levels]
        plt.boxplot(box_plot_data, labels=levels)
        plt.ylabel("Score")
        plt.savefig(f'./boxplots/{metric}_levels.png')

    # Algorithm boxplots
    for metric in ['VR', 'VFA', 'RPA', 'RCA', 'OA']:
        fig = plt.figure()
        fig.suptitle(metric, fontsize=14, fontweight='bold')
        box_plot_data = [results[metric][algorithm] for algorithm in algorithms]
        plt.boxplot(box_plot_data, labels=algorithms)
        plt.ylabel("Score")
        plt.savefig(f'./boxplots/{metric}_algorithms.png')
        
        
def save_specific_boxplots(specific_results):
    levels = ['LEVEL1', 'LEVEL2', 'LEVEL3']
    algorithms = ['Yin', 'YinFFT', 'CREPE']
    
    for metric in ['VR', 'VFA', 'RPA', 'RCA', 'OA']:
        for level in levels:
            fig = plt.figure()
            fig.suptitle(f'{metric}, {level}', fontsize=14, fontweight='bold')
            box_plot_data = [
                specific_results[metric][level][algorithm]
                for algorithm in algorithms
            ]
            plt.boxplot(box_plot_data, labels=algorithms)
            plt.ylabel('Score')
            plt.savefig(f'./boxplots/{metric}_{level}.png')
            plt.close()

    for metric in ['VR', 'VFA', 'RPA', 'RCA', 'OA']:
        for algorithm in algorithms:
            fig = plt.figure()
            fig.suptitle(f'{metric}, {algorithm}', fontsize=14, fontweight='bold')
            box_plot_data = [
                specific_results[metric][level][algorithm]
                for level in levels
            ]
            plt.boxplot(box_plot_data, labels=levels)
            plt.ylabel('Score')
            plt.savefig(f'./boxplots/{metric}_{algorithm}.png')
            plt.close()
            
            
def save_barplot(specific_results):
    shutil.rmtree('./barplots', ignore_errors=True)
    os.mkdir('./barplots')
    
    dictionary = {
        'StaccatoEb': ['LEVEL1', 0, 24], 'LegatoBb': ['LEVEL1', 1, 24],
        'LegatoC': ['LEVEL1', 2, 24], 'StaccatoAb': ['LEVEL1', 3, 24],
        'StaccatoG': ['LEVEL1', 4, 24], 'StaccatoD': ['LEVEL1', 5, 24],
        'LegatoD': ['LEVEL1', 6, 24], 'StaccatoC': ['LEVEL1', 7, 24],
        'LegatoF#': ['LEVEL1', 8, 24], 'StaccatoF': ['LEVEL1', 9, 24],
        'StaccatoC#': ['LEVEL1', 10, 24], 'LegatoB': ['LEVEL1', 11, 24],
        'StaccatoBb': ['LEVEL1', 12, 24], 'LegatoF': ['LEVEL1', 13, 24],
        'StaccatoB': ['LEVEL1', 14, 24], 'StaccatoA': ['LEVEL1', 15, 24],
        'LegatoEb': ['LEVEL1', 16, 24], 'LegatoG': ['LEVEL1', 17, 24],
        'LegatoC#': ['LEVEL1', 18, 24], 'LegatoAb': ['LEVEL1', 19, 24],
        'StaccatoE': ['LEVEL1', 20, 24], 'StaccatoF#': ['LEVEL1', 21, 24],
        'LegatoA': ['LEVEL1', 22, 24], 'LegatoE': ['LEVEL1', 23, 24],
        'myIdeal': ['LEVEL2', 0, 6], 'turnaround': ['LEVEL2', 1, 6],
        'GoodbyePorkPieHat': ['LEVEL2', 2, 6], 'secretlove': ['LEVEL2', 3, 6],
        'SaborAmi': ['LEVEL2', 4, 6], 'timeWasMelo': ['LEVEL2', 5, 6],
        'ColtraneSolo2ndTimeWas': ['LEVEL3', 0, 8], 'donnaLee': ['LEVEL3', 1, 8],
        'zyryab': ['LEVEL3', 2, 8], 'ColtraneSolo1stTimeWas': ['LEVEL3', 3, 8],
        'lickTimeWasCompo': ['LEVEL3', 4, 8], 'ColtraneSolo4thTimeWas': ['LEVEL3', 5, 8],
        'ColtraneSolo3rdTimeWas': ['LEVEL3', 6, 8], 'airegin': ['LEVEL3', 7, 8],
    }
    
    categories = ['Yin', 'YinFFT', 'CREPE']
    n_categories = len(categories)
    width = 0.15
    
    for audio in dictionary.keys():
        VR = [
            np.mean(specific_results['VR'][dictionary[audio][0]]['Yin'][dictionary[audio][1]::dictionary[audio][2]]),
            np.mean(specific_results['VR'][dictionary[audio][0]]['YinFFT'][dictionary[audio][1]::dictionary[audio][2]]),
            np.mean(specific_results['VR'][dictionary[audio][0]]['CREPE'][dictionary[audio][1]::dictionary[audio][2]])
        ]
        VFA = [
            np.mean(specific_results['VFA'][dictionary[audio][0]]['Yin'][dictionary[audio][1]::dictionary[audio][2]]),
            np.mean(specific_results['VFA'][dictionary[audio][0]]['YinFFT'][dictionary[audio][1]::dictionary[audio][2]]),
            np.mean(specific_results['VFA'][dictionary[audio][0]]['CREPE'][dictionary[audio][1]::dictionary[audio][2]])
        ]
        RPA = [
            np.mean(specific_results['RPA'][dictionary[audio][0]]['Yin'][dictionary[audio][1]::dictionary[audio][2]]),
            np.mean(specific_results['RPA'][dictionary[audio][0]]['YinFFT'][dictionary[audio][1]::dictionary[audio][2]]),
            np.mean(specific_results['RPA'][dictionary[audio][0]]['CREPE'][dictionary[audio][1]::dictionary[audio][2]])
        ]
        RCA = [
            np.mean(specific_results['RCA'][dictionary[audio][0]]['Yin'][dictionary[audio][1]::dictionary[audio][2]]),
            np.mean(specific_results['RCA'][dictionary[audio][0]]['YinFFT'][dictionary[audio][1]::dictionary[audio][2]]),
            np.mean(specific_results['RCA'][dictionary[audio][0]]['CREPE'][dictionary[audio][1]::dictionary[audio][2]])
        ]
        OA = [
            np.mean(specific_results['OA'][dictionary[audio][0]]['Yin'][dictionary[audio][1]::dictionary[audio][2]]),
            np.mean(specific_results['OA'][dictionary[audio][0]]['YinFFT'][dictionary[audio][1]::dictionary[audio][2]]),
            np.mean(specific_results['OA'][dictionary[audio][0]]['CREPE'][dictionary[audio][1]::dictionary[audio][2]])
        ]

        ind = np.arange(n_categories)

        fig, ax = plt.subplots()

        ax.bar(ind - 2 * width, VR, width, label='VR', color='skyblue')
        ax.bar(ind - width, VFA, width, label='VFA', color='lightgreen')
        ax.bar(ind, RPA, width, label='RPA', color='salmon')
        ax.bar(ind + width, RCA, width, label='RCA', color='orange')
        ax.bar(ind + 2 * width, OA, width, label='OA', color='purple')

        ax.set_title(audio)
        ax.set_xlabel('Algorithms')
        ax.set_ylabel('Score')
        ax.set_xticks(ind)
        ax.set_xticklabels(categories)
        ax.legend()
    
        plt.savefig(f'./barplots/{audio}_metrics.png')
        plt.close()
        
        
def print_interesting_stats(specific_results):
    levels = ['LEVEL1', 'LEVEL2', 'LEVEL3']
    algorithms = ['Yin', 'YinFFT', 'CREPE']
    metrics = ['VR', 'VFA', 'RPA', 'RCA', 'OA']
    
    for metric in metrics:
        for level in levels:
            for algorithm in algorithms:
                # Interesting stats (specific boxplots)
                print(f'{metric}_{level}_{algorithm} stats:')
                print("----- max: ", np.max(specific_results[metric][level][algorithm]))
                print("----- min: ", np.min(specific_results[metric][level][algorithm]))
                print("----- mean: ", np.mean(specific_results[metric][level][algorithm]))
                print("")


def main():
    results = {
        'VR': {'LEVEL1': [], 'LEVEL2': [], 'LEVEL3': [], 'Yin': [],
               'YinFFT': [], 'CREPE': []},
        'VFA': {'LEVEL1': [], 'LEVEL2': [], 'LEVEL3': [], 'Yin': [],
                'YinFFT': [], 'CREPE': []},
        'RPA': {'LEVEL1': [], 'LEVEL2': [], 'LEVEL3': [], 'Yin': [],
                'YinFFT': [], 'CREPE': []},
        'RCA': {'LEVEL1': [], 'LEVEL2': [], 'LEVEL3': [], 'Yin': [],
                'YinFFT': [], 'CREPE': []},
        'OA': {'LEVEL1': [], 'LEVEL2': [], 'LEVEL3': [], 'Yin': [],
               'YinFFT': [], 'CREPE': []},
    }
    
    specific_results = {
        'VR': {'LEVEL1': {'Yin': [], 'YinFFT': [], 'CREPE': []},
               'LEVEL2': {'Yin': [], 'YinFFT': [], 'CREPE': []},
               'LEVEL3': {'Yin': [], 'YinFFT': [], 'CREPE': []}},
        'VFA': {'LEVEL1': {'Yin': [], 'YinFFT': [], 'CREPE': []},
                'LEVEL2': {'Yin': [], 'YinFFT': [], 'CREPE': []},
                'LEVEL3': {'Yin': [], 'YinFFT': [], 'CREPE': []}},
        'RPA': {'LEVEL1': {'Yin': [], 'YinFFT': [], 'CREPE': []},
                'LEVEL2': {'Yin': [], 'YinFFT': [], 'CREPE': []},
                'LEVEL3': {'Yin': [], 'YinFFT': [], 'CREPE': []}},
        'RCA': {'LEVEL1': {'Yin': [], 'YinFFT': [], 'CREPE': []},
                'LEVEL2': {'Yin': [], 'YinFFT': [], 'CREPE': []},
                'LEVEL3': {'Yin': [], 'YinFFT': [], 'CREPE': []}},
        'OA': {'LEVEL1': {'Yin': [], 'YinFFT': [], 'CREPE': []},
               'LEVEL2': {'Yin': [], 'YinFFT': [], 'CREPE': []},
               'LEVEL3': {'Yin': [], 'YinFFT': [], 'CREPE': []}},
    }

    for folder in os.listdir('./RESULTS'):
        print(f'Analysing {folder} results ...')
        path = os.path.join('./RESULTS', folder)
        shutil.rmtree(os.path.join(path, 'metrics'), ignore_errors=True)
        os.mkdir(os.path.join(path, 'metrics'))

        analyze_folder(path, folder, results, specific_results)
    
    print("Saving metrics boxplots ...")
    save_boxplots(results)
    save_specific_boxplots(specific_results)
    print("Saving metrics barplots ...")
    save_barplot(specific_results)
    print("")
    print_interesting_stats(specific_results)


if __name__ == '__main__':
    main()
