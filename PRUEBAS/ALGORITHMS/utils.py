import os
import essentia.standard as es
import essentia as ES
from pylab import plot
import matplotlib.pyplot as plt
import numpy as np

def save_values_csv(path, audioName, array):
    np.savetxt(str(path)+'/values/'+audioName+'_values.csv', array)
    
def save_confidence_csv(path, audioName, array):
    np.savetxt(str(path)+'/confidence/'+audioName+'_confidence.csv', array)

def save_times_csv(path, audioName, array):
    np.savetxt(str(path)+'/times/'+audioName+'_times.csv', array)
    
def save_all_csv(path, audioName, array1, array2, array3):
    save_values_csv(path, audioName, array1)
    save_confidence_csv(path, audioName, array2)
    save_times_csv(path, audioName, array3)
    
def save_midi_csv(path, audioName, notes, onsets, durations):
    np.savetxt(str(path)+'/midi/'+audioName+'_estimated__midinotes.csv', notes)
    np.savetxt(str(path)+'/midi/'+audioName+'_estimated_midionsets.csv', onsets)
    np.savetxt(str(path)+'/midi/'+audioName+'_estimated_mididurations.csv', durations)

def list_audiofile_names(level):
    audiofile_names = []    
    for file in os.listdir('./Audio/LEVEL'+str(level)+'/WAV'):
        audiofile_names.append(file[:-4])
    return audiofile_names

def list_audiofiles(audiofile_names, level):
    audiofiles = []
    for i in audiofile_names:
        audiofile = './Audio/LEVEL'+str(level)+'/WAV/'+i+'.wav'
        audiofiles.append(audiofile)  
    return audiofiles

def YinComputation(audio, frameSize, hopSize, pitch_extractor):
    pitch_values = []
    pitch_confidence = []
    for frame in es.FrameGenerator(audio, frameSize=frameSize, hopSize=hopSize, startFromZero=True):
        frequency, confidence = pitch_extractor(frame)
        pitch_values.append(frequency)
        pitch_confidence.append(confidence)
        
    pitch_values = ES.array(pitch_values).T
    pitch_confidence = ES.array(pitch_confidence).T
    pitch_times = np.linspace(0.0,len(audio)/44100.0,len(pitch_values))
    return pitch_values, pitch_confidence, pitch_times

def YinFFTComputation(audio, frameSize, hopSize, pitch_extractor, w, spectrum):
    pitch_values = []
    pitch_confidence = []
    for frame in es.FrameGenerator(audio, frameSize=frameSize, hopSize=hopSize, startFromZero=True):
        frequency, confidence = pitch_extractor(spectrum(w(frame)))
        pitch_values.append(frequency)
        pitch_confidence.append(confidence)
        
    pitch_values = ES.array(pitch_values).T
    pitch_confidence = ES.array(pitch_confidence).T
    pitch_times = np.linspace(0.0,len(audio)/44100.0,len(pitch_values))
    return pitch_values, pitch_confidence, pitch_times

def askAboutCREPE():
    print("Select PitchCREPE's Graph:")
    print('---- 1. Tiny')
    print('---- 2. Small')
    print('---- 3. Medium')
    print('---- 4. Large')
    print('---- 5. Full')
    graph = int(input())
    return graph

def decideGraphCREPE(graph, hopSize):
    if (graph == 1):
        str_algorithm = 'PitchCREPEtiny'
        pitch_extractor = es.PitchCREPE(graphFilename="./CREPE/crepe-tiny-1.pb", hopSize=hopSize)
    elif (graph == 2):
        str_algorithm = 'PitchCREPEsmall'
        pitch_extractor = es.PitchCREPE(graphFilename="./CREPE/crepe-small-1.pb", hopSize=hopSize)
    elif (graph == 3):
        str_algorithm = 'PitchCREPEmedium'
        pitch_extractor = es.PitchCREPE(graphFilename="./CREPE/crepe-medium-1.pb", hopSize=hopSize)
    elif (graph == 4):
        str_algorithm = 'PitchCREPElarge'
        pitch_extractor = es.PitchCREPE(graphFilename="./CREPE/crepe-large-1.pb", hopSize=hopSize)
    else:
        str_algorithm = 'PitchCREPEfull'
        pitch_extractor = es.PitchCREPE(graphFilename="./CREPE/crepe-full-1.pb", hopSize=hopSize)
        
    return str_algorithm, pitch_extractor

def mkdirResults(level, str_algorithm, str_loader, sampleRate, frameSize, hopSize):
    path = './RESULTS/LEVEL'+str(level)+str_algorithm+str_loader+'_'+str(sampleRate)+'_'+str(frameSize)+'_'+str(hopSize)
    os.mkdir(path)
    os.mkdir(path+'/pitchplots')
    os.mkdir(path+'/audioplots')
    os.mkdir(path+'/values')
    os.mkdir(path+'/confidence')
    os.mkdir(path+'/times')
    os.mkdir(path+'/midi')
    return path

def plotAudio(audio, sampleRate, path, audioName):
    plot(audio[0*sampleRate:int(len(audio)/float(sampleRate))*sampleRate])
    plt.title("This is how "+audioName+" looks like:")
    plt.savefig(str(path)+'/audioplots/'+audioName+'.png')

def plot_estimated_pitch(pitch_times, pitch_values, pitch_confidence, path, audioName):
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(pitch_times, pitch_values)
    axarr[0].set_title('estimated pitch [Hz]')
    axarr[1].plot(pitch_times, pitch_confidence)
    axarr[1].set_title('pitch confidence')
    plt.savefig(str(path)+'/pitchplots/'+audioName+'_estimatedPitch.png')
    
def convert_timestamps(pitch_times, pitch_values, notes, onsets, durations):
    offsets = onsets + durations # finales de notas
    ref_times = pitch_times
    ref_values = np.zeros(len(ref_times))
    j=0
    for i in ref_times:
        if (i >= onsets[0]):
            if (i < offsets[0]):
                ref_values[j] = notes[0]
            else:
                # comprobar que no haya notas seguidas
                if (len(onsets) > 1):
                    if (i >= onsets[1]) and (i < offsets[1]):
                        ref_values[j] = notes[1]
                        offsets = np.delete(offsets, 0)
                        onsets = np.delete(onsets, 0)
                        notes = np.delete(notes, 0)
                    else:
                        offsets = np.delete(offsets, 0)
                        onsets = np.delete(onsets, 0)
                        notes = np.delete(notes, 0)
                
        j=j+1
    return ref_times, ref_values
    
    
