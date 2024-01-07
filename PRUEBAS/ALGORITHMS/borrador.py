# borrador

import utils
import numpy as np
import essentia.standard as es
import essentia as ES
import sys
import os
from pylab import plot
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (15, 6)

try:
    level = int(sys.argv[1])
except:
    print("Select level")
    sys.exit()

print('Select Pitch Algorithm:')
print('---- 1. PitchYin')
print('---- 2. PitchYinFFT')
print('---- 3. PitchCREPE')
algorithm = int(input())

print('Select Audio Loader:')
print('---- 1. EqloudLoader')
print('---- 2. EasyLoader')
print('---- 3. MonoLoader')
print('---- 4. AudioLoader')
loader_type = int(input())
if algorithm==3 and loader_type==4:
    print("Error: PitchCREPE Algorithm and AudioLoader function are incompatible")
    sys.exit()

print('Select sampleRate (Hz):')
print('---- 16000, 32000, 44100, 48000, ...')
sampleRate = int(input())
if algorithm==3 and sampleRate!=16000:
    print("Error: PitchCREPE Algorithm must use 16 kHz as sampleRate")
    sys.exit()
if loader_type==4 and sampleRate!=44100:
    print("Error: AudioLoader only accepts 44100 Hz as sampleRate")
    sys.exit()
    
print('Select frameSize:')
print('---- 0, 1024, 2048, ...(for the CREPE algorithm it is not necessary, select 0)')
frameSize = int(input())

print('Select hopSize:')
print('---- 10, 128, 512, ... (for the CREPE algorithm 10 is recommended)')
hopSize = int(input())
"""
audiofiles =[]    
for file in os.listdir('./Audio/LEVEL'+str(level)+'/WAV'):
    audiofiles.append(file[:-4])
"""
audiofile_names = utils.list_audiofile_names(level)
audiofiles = utils.list_audiofiles(audiofile_names, level)
"""
audiofiles = []
for i in audiofile_names:
    audiofile = './Audio/LEVEL'+str(level)+'/WAV/'+i+'.wav'
    audiofiles.append(audiofile)
"""
if (loader_type == 1):
    str_loader = 'EqLoud'
    loader = es.EqloudLoader(filename=audiofile, sampleRate=sampleRate)
    audio = loader()
elif (loader_type == 2):
    str_loader = 'Easy'
    loader = es.EasyLoader(filename=audiofile, sampleRate=sampleRate)
    audio = loader()
elif (loader_type == 3):
    str_loader = 'Mono'
    loader = es.MonoLoader(filename=audiofile, sampleRate=sampleRate)
    audio = loader()
else:
    str_loader = 'Audio'
    loader = es.AudioLoader(filename=audiofile)
    audio_vector = loader()
    audio = audio_vector[0]
    sampleRate = audio_vector[1]


if (algorithm == 1):
    str_algorithm = 'PitchYin'
    pitch_extractor = es.PitchYin(frameSize=frameSize, sampleRate=sampleRate)
    pitch_values, pitch_confidence, pitch_times = utils.YinComputation(audio, frameSize, hopSize, pitch_extractor)
    """
    pitch_values = []
    pitch_confidence = []
    for frame in es.FrameGenerator(audio, frameSize=frameSize, hopSize=hopSize, startFromZero=True):
        frequency, confidence = pitch_extractor(frame)
        pitch_values.append(frequency)
        pitch_confidence.append(confidence)
        
    pitch_values = ES.array(pitch_values).T
    pitch_confidence = ES.array(pitch_confidence).T
    pitch_times = np.linspace(0.0,len(audio)/44100.0,len(pitch_values))
    """
elif (algorithm == 2):
    str_algorithm = 'PitchYinFFT'
    pitch_extractor = es.PitchYinFFT(frameSize=frameSize, sampleRate=sampleRate)
    w = es.Windowing(type = 'hann')
    spectrum = es.Spectrum()
    pitch_values, pitch_confidence, pitch_times = utils.YinFFTComputation(audio, frameSize, hopSize, pitch_extractor, w, spectrum)
    """
    pitch_values = []
    pitch_confidence = []
    for frame in es.FrameGenerator(audio, frameSize=frameSize, hopSize=hopSize, startFromZero=True):
        frequency, confidence = pitch_extractor(spectrum(w(frame)))
        pitch_values.append(frequency)
        pitch_confidence.append(confidence)
        
    pitch_values = ES.array(pitch_values).T
    pitch_confidence = ES.array(pitch_confidence).T
    pitch_times = np.linspace(0.0,len(audio)/44100.0,len(pitch_values))
    """
else:
    sampleRate = 16000
    frameSize = 0
    graph = utils.askAboutCREPE()
    str_algorithm, pitch_extractor = utils.decideGraphCREPE(graph, hopSize)
    pitch_times, pitch_values, pitch_confidence, pitch_activations = pitch_extractor(audio) # 4 = i
    """
    sampleRate = 16000
    frameSize = 0
    print("Select PitchCREPE's Graph:")
    print('---- 1. Tiny')
    print('---- 2. Small')
    print('---- 3. Medium')
    print('---- 4. Large')
    print('---- 5. Full')
    graph = int(input())
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
    
    pitch_times, pitch_values, pitch_confidence, pitch_activations = pitch_extractor(audio)
    """
    
print("Duration of the audio sample [sec]:")
print(len(audio)/float(sampleRate)) # 4 = i

path = utils.mkdirResults(level, str_algorithm, str_loader, sampleRate, frameSize, hopSize)
"""
path = './RESULTS/LEVEL'+str(level)+str_algorithm+str_loader+'_'+str(sampleRate)+'_'+str(frameSize)+'_'+str(hopSize)
os.mkdir(path)
"""
utils.plotAudio(audio, sampleRate, path, audiofile_names[4]) # 4 = i
"""
plot(audio[0*sampleRate:int(len(audio)/float(sampleRate))*sampleRate])
plt.title("This is how "+audiofiles[4]+" looks like:") # 4=i
plt.savefig(str(path)+'/'+audiofiles[4]+'.png') # 4=i
"""
utils.plot_estimated_pitch(pitch_times, pitch_values, pitch_confidence, path, audiofile_names[4]) # 4 = i
"""
f, axarr = plt.subplots(2, sharex=True)
axarr[0].plot(pitch_times, pitch_values)
axarr[0].set_title('estimated pitch [Hz]')
axarr[1].plot(pitch_times, pitch_confidence)
axarr[1].set_title('pitch confidence')
plt.savefig(str(path)+'/'+audiofiles[4]+'_estimatedPitch.png') # 4=i
"""


# métricas de evaluación de la librería mir_eval, next thing to do

