"""
mir_eval implementation

"""
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import mir_eval.melody as mir
import utilsPEP


def analyze_folder(path, folder, results):
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

        pitch_times = utilsPEP.extract_csv(os.path.join(path, 'times', f'{audioname}_times.csv'))
        pitch_values = utilsPEP.extract_csv(os.path.join(path, 'values', f'{audioname}_values.csv'))
        pitch_confidence = utilsPEP.extract_csv(os.path.join(path, 'confidence', f'{audioname}_confidence.csv'))
        notes = utilsPEP.extract_csv(os.path.join('./Audio', level, 'CSV', f'{audioname}_midinotes.csv'))
        onsets = utilsPEP.extract_csv(os.path.join('./Audio', level, 'CSV', f'{audioname}_midionsets.csv'))
        durations = utilsPEP.extract_csv(os.path.join('./Audio', level, 'CSV', f'{audioname}_mididurations.csv'))
        ref_times, ref_values = utilsPEP.convert_timestamps(pitch_times, pitch_values, notes, onsets, durations)

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
    utilsPEP.save_metrics_csv(os.path.join(path, 'metrics', 'VR.csv'), voicing_recall)
    utilsPEP.save_metrics_csv(os.path.join(path, 'metrics', 'VFA.csv'), voicing_false_alarm)
    utilsPEP.save_metrics_csv(os.path.join(path, 'metrics', 'RPA.csv'), raw_pitch_accuracy)
    utilsPEP.save_metrics_csv(os.path.join(path, 'metrics', 'RCA.csv'), raw_chroma_accuracy)
    utilsPEP.save_metrics_csv(os.path.join(path, 'metrics', 'OA.csv'), overall_accuracy)

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


def main():
    results = {
        'VR': {'LEVEL1': [], 'LEVEL2': [], 'LEVEL3': [], 'Yin': [], 'YinFFT': [], 'CREPE': []},
        'VFA': {'LEVEL1': [], 'LEVEL2': [], 'LEVEL3': [], 'Yin': [], 'YinFFT': [], 'CREPE': []},
        'RPA': {'LEVEL1': [], 'LEVEL2': [], 'LEVEL3': [], 'Yin': [], 'YinFFT': [], 'CREPE': []},
        'RCA': {'LEVEL1': [], 'LEVEL2': [], 'LEVEL3': [], 'Yin': [], 'YinFFT': [], 'CREPE': []},
        'OA': {'LEVEL1': [], 'LEVEL2': [], 'LEVEL3': [], 'Yin': [], 'YinFFT': [], 'CREPE': []},
    }

    for folder in os.listdir('./RESULTS'):
        print(f'Analysing {folder} results ...')
        path = os.path.join('./RESULTS', folder)
        shutil.rmtree(os.path.join(path, 'metrics'), ignore_errors=True)
        os.mkdir(os.path.join(path, 'metrics'))

        analyze_folder(path, folder, results)

    print("Saving metrics boxplots ...")
    save_boxplots(results)


if __name__ == '__main__':
    main()
