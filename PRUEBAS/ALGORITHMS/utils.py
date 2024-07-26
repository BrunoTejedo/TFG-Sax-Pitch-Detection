import os
import essentia.standard as es
import essentia as ES
from pylab import plot
import matplotlib.pyplot as plt
import numpy as np
import csv


def save_values_csv(path, audio_name, array):
    np.savetxt(f'{path}/values/{audio_name}_values.csv', array)


def save_confidence_csv(path, audio_name, array):
    np.savetxt(f'{path}/confidence/{audio_name}_confidence.csv', array)


def save_times_csv(path, audio_name, array):
    np.savetxt(f'{path}/times/{audio_name}_times.csv', array)


def save_all_csv(path, audio_name, array1, array2, array3):
    save_values_csv(path, audio_name, array1)
    save_confidence_csv(path, audio_name, array2)
    save_times_csv(path, audio_name, array3)


def save_midi_csv(path, audio_name, notes, onsets, durations):
    np.savetxt(f'{path}/midi/{audio_name}_estimated_midinotes.csv', notes)
    np.savetxt(f'{path}/midi/{audio_name}_estimated_midionsets.csv', onsets)
    np.savetxt(f'{path}/midi/{audio_name}_estimated_mididurations.csv', durations)


def list_audiofile_names(level):
    audiofile_names = []
    for file in os.listdir(f'./Audio/LEVEL{level}/WAV'):
        audiofile_names.append(file[:-4])
    return audiofile_names


def list_audiofiles(audiofile_names, level):
    audiofiles = []
    for name in audiofile_names:
        audiofile = f'./Audio/LEVEL{level}/WAV/{name}.wav'
        audiofiles.append(audiofile)
    return audiofiles


def yin_computation(audio, frame_size, hop_size, pitch_extractor):
    pitch_values = []
    pitch_confidence = []
    for frame in es.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size, startFromZero=True):
        frequency, confidence = pitch_extractor(frame)
        pitch_values.append(frequency)
        pitch_confidence.append(confidence)

    pitch_values = ES.array(pitch_values).T
    pitch_confidence = ES.array(pitch_confidence).T
    pitch_times = np.linspace(0.0, len(audio) / 44100.0, len(pitch_values))
    return pitch_values, pitch_confidence, pitch_times


def yin_fft_computation(audio, frame_size, hop_size, pitch_extractor, w, spectrum):
    pitch_values = []
    pitch_confidence = []
    for frame in es.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size, startFromZero=True):
        frequency, confidence = pitch_extractor(spectrum(w(frame)))
        pitch_values.append(frequency)
        pitch_confidence.append(confidence)

    pitch_values = ES.array(pitch_values).T
    pitch_confidence = ES.array(pitch_confidence).T
    pitch_times = np.linspace(0.0, len(audio) / 44100.0, len(pitch_values))
    return pitch_values, pitch_confidence, pitch_times


def ask_about_crepe():
    print("Select PitchCREPE's Graph:")
    print('---- 1. Tiny')
    print('---- 2. Small')
    print('---- 3. Medium')
    print('---- 4. Large')
    print('---- 5. Full')
    graph = int(input())
    return graph


def decide_graph_crepe(graph, hop_size):
    if graph == 1:
        str_algorithm = 'PitchCREPEtiny'
        pitch_extractor = es.PitchCREPE(graphFilename="./CREPE/crepe-tiny-1.pb", hopSize=hop_size)
    elif graph == 2:
        str_algorithm = 'PitchCREPEsmall'
        pitch_extractor = es.PitchCREPE(graphFilename="./CREPE/crepe-small-1.pb", hopSize=hop_size)
    elif graph == 3:
        str_algorithm = 'PitchCREPEmedium'
        pitch_extractor = es.PitchCREPE(graphFilename="./CREPE/crepe-medium-1.pb", hopSize=hop_size)
    elif graph == 4:
        str_algorithm = 'PitchCREPElarge'
        pitch_extractor = es.PitchCREPE(graphFilename="./CREPE/crepe-large-1.pb", hopSize=hop_size)
    else:
        str_algorithm = 'PitchCREPEfull'
        pitch_extractor = es.PitchCREPE(graphFilename="./CREPE/crepe-full-1.pb", hopSize=hop_size)

    return str_algorithm, pitch_extractor


def mkdir_results(level, str_algorithm, str_loader, sample_rate, frame_size, hop_size):
    path = f'./RESULTS/LEVEL{level}{str_algorithm}{str_loader}_{sample_rate}_{frame_size}_{hop_size}'
    os.mkdir(path)
    os.mkdir(f'{path}/pitchplots')
    os.mkdir(f'{path}/audioplots')
    os.mkdir(f'{path}/values')
    os.mkdir(f'{path}/confidence')
    os.mkdir(f'{path}/times')
    os.mkdir(f'{path}/midi')
    return path


def plot_audio(audio, sample_rate, path, audio_name):
    plt.clf()
    plot(audio[0 * sample_rate:int(len(audio) / float(sample_rate)) * sample_rate])
    plt.title(f"This is how {audio_name} looks like:")
    plt.xlabel("Audio samples")
    plt.ylabel("Normalized amplitude")
    plt.savefig(f'{path}/audioplots/{audio_name}.png')
    plt.close()


def plot_estimated_pitch(pitch_times, pitch_values, pitch_confidence, path, audio_name):
    plt.clf()
    
    f, axarr = plt.subplots(2, sharex=True)
    
    axarr[0].plot(pitch_times, pitch_values)
    axarr[0].set_title('Estimated Pitch [Hz]')
    axarr[0].set_ylabel('Pitch [Hz]')
    
    axarr[1].plot(pitch_times, pitch_confidence)
    axarr[1].set_title('Pitch Confidence')
    axarr[1].set_xlabel('Time [s]')
    axarr[1].set_ylabel('Confidence')
    
    plt.savefig(f'{path}/pitchplots/{audio_name}_estimatedPitch.png')
    plt.close()


def convert_timestamps(pitch_times, pitch_values, notes, onsets, durations):
    offsets = onsets + durations  # notes endings
    ref_times = pitch_times
    ref_values = np.zeros(len(ref_times))
    j = 0
    for i in ref_times:
        if i >= onsets[0]:
            if i < offsets[0]:
                ref_values[j] = notes[0]
            else:
                # check that there are no consecutive notes
                if len(onsets) > 1:
                    if i >= onsets[1] and i < offsets[1]:
                        ref_values[j] = notes[1]
                        offsets = np.delete(offsets, 0)
                        onsets = np.delete(onsets, 0)
                        notes = np.delete(notes, 0)
                    else:
                        offsets = np.delete(offsets, 0)
                        onsets = np.delete(onsets, 0)
                        notes = np.delete(notes, 0)
        j += 1
    return ref_times, ref_values


def extract_csv(path):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)

    data_array = np.array(data)
    data_array = np.array(data, dtype=float)
    data_array = np.transpose(data_array)
    return data_array[0]


def save_metrics_csv(path, array):
    np.savetxt(str(path), array)
