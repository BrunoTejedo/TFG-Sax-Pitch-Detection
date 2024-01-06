"""
I want to use these 3 algorithms:

- PitchCREPE
- PitchYin
- PitchYinFFT

"""

from pylab import plot, show, figure, imshow
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (15, 6)

import numpy as np
import essentia.standard as es
import essentia as ES
import sys, csv
import os

try:
    level = int(sys.argv[1])
except:
    print("Select level")
    sys.exit()

audiofiles =[]    
for file in os.listdir('./Audio/LEVEL'+str(level)+'/WAV'):
    audiofiles.append(file[:-4])

audiofile = './Audio/LEVEL'+str(level)+'/WAV/'+audiofiles[4]+'.wav' # 4=i

print('Select Pitch Algorithm:')
print('---- 1. PitchYin') # patience (slowest)
print('---- 2. PitchYinFFT') # windowing, spectrum, etc.
print('---- 3. PitchCREPE') # caso especial (sampleRate=16000, tensorflow, etc.)
algorithm = int(input())

print('Select Audio Loader:')
print('---- 1. EqloudLoader')
print('---- 2. EasyLoader')
print('---- 3. MonoLoader')
print('---- 4. AudioLoader') # caso especial
loader_type = int(input())

print('Select sampleRate (Hz):')
print('---- 16000, 32000, 44100, 48000, ...')
sampleRate = int(input())

if (loader_type == 1):
    str_loader = 'EqLoud'
    loader = es.EqloudLoader(filename=audiofile, sampleRate=sampleRate)
elif (loader_type == 2):
    str_loader = 'Easy'
    loader = es.EasyLoader(filename=audiofile, sampleRate=sampleRate)
elif (loader_type == 3):
    str_loader = 'Mono'
    loader = es.MonoLoader(filename=audiofile, sampleRate=sampleRate)
else:
    str_loader = 'Audio' # caso especial
    loader = es.AudioLoader(filename=audiofile)
    
print('Select frameSize:')
print('---- 1024, 2048, ...')
frameSize = int(input())

print('Select hopSize:')
print('---- 10, 128, 512, ...')
hopSize = int(input())

audio = loader()
print("Duration of the audio sample [sec]:")
print(len(audio)/float(sampleRate))

if (algorithm == 1):
    str_algorithm = 'PitchYin'
    pitch_values = []
    pitch_confidence = []
    pitch_extractor = es.PitchYin(frameSize=frameSize, sampleRate=sampleRate)
    for frame in es.FrameGenerator(audio, frameSize=frameSize, hopSize=hopSize, startFromZero=True):
        frequency, confidence = pitch_extractor(frame)
        pitch_values.append(frequency)
        pitch_confidence.append(confidence)
        
    pitch_values = ES.array(pitch_values).T
    pitch_confidence = ES.array(pitch_confidence).T
    pitch_times = np.linspace(0.0,len(audio)/44100.0,len(pitch_values))
elif (algorithm == 2):
    str_algorithm = 'PitchYinFFT'
    w = es.Windowing(type = 'hann')
    spectrum = es.Spectrum()
    pitch_values = []
    pitch_confidence = []
    pitch_extractor = es.PitchYinFFT(frameSize=frameSize, sampleRate=sampleRate)
    for frame in es.FrameGenerator(audio, frameSize=frameSize, hopSize=hopSize, startFromZero=True):
        frequency, confidence = pitch_extractor(spectrum(w(frame)))
        pitch_values.append(frequency)
        pitch_confidence.append(confidence)
        
    pitch_values = ES.array(pitch_values).T
    pitch_confidence = ES.array(pitch_confidence).T
    pitch_times = np.linspace(0.0,len(audio)/44100.0,len(pitch_values))
else:
    str_algorithm = 'PitchCREPE'

path = './RESULTS/LEVEL'+str(level)+str_algorithm+str_loader+'_'+str(sampleRate)+'_'+str(frameSize)+'_'+str(hopSize)
os.mkdir(path)

plot(audio[0*sampleRate:int(len(audio)/float(sampleRate))*sampleRate])
plt.title("This is how "+audiofiles[4]+" looks like:") # 4=i
plt.savefig(str(path)+'/'+audiofiles[4]+'.png') # 4=i

# Plot the estimated pitch and confidence over time
f, axarr = plt.subplots(2, sharex=True)
axarr[0].plot(pitch_times, pitch_values)
axarr[0].set_title('estimated pitch [Hz]')
axarr[1].plot(pitch_times, pitch_confidence)
axarr[1].set_title('pitch confidence')
plt.savefig(str(path)+'/'+audiofiles[4]+'_estimatedPitch.png') # 4=i

dsinikndo

"""
# 1) PitchCREPE (RuntimeError: In PitchCREPE.compute: TensorflowPredict: This algorithm 
is not configured. To configure this algorithm you should specify a valid `graphFilename` 
or `savedModel` as input parameter.)
# ATENCIÓN: The required sample rate of input signal is 16 KHz (modificar arriba)
pitch_extractor = es.PitchCREPE(hopSize=128) #hopSize?
times, pitch_values, pitch_confidence, activations = pitch_extractor(audio)
"""
# hasta aquí ya tengo trabajo, cuando tenga esto sigo adelante con las métricas de 
# evaluación de la librería mir_eval, los archivos midi y tota la pesca.
