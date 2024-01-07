import os
import essentia.standard as es
import essentia as ES
from pylab import plot
import matplotlib.pyplot as plt
import numpy as np

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
    return path

def plotAudio(audio, sampleRate, path, audioName):
    plot(audio[0*sampleRate:int(len(audio)/float(sampleRate))*sampleRate])
    plt.title("This is how "+audioName+" looks like:")
    plt.savefig(str(path)+'/'+audioName+'.png')

def plot_estimated_pitch(pitch_times, pitch_values, pitch_confidence, path, audioName):
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(pitch_times, pitch_values)
    axarr[0].set_title('estimated pitch [Hz]')
    axarr[1].plot(pitch_times, pitch_confidence)
    axarr[1].set_title('pitch confidence')
    plt.savefig(str(path)+'/'+audioName+'_estimatedPitch.png')