# esto será el utils

# pruebas (CREPE n AudioLoader)

from pylab import plot, show, figure, imshow
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (15, 6)

import numpy as np
import essentia.standard as es
import essentia as ES
import sys, csv
import os

audiofile = './Audio/LEVEL1/WAV/StaccatoF.wav'

loader = es.EqloudLoader(filename=audiofile, sampleRate=44100)

audio = loader()

print("Duration of StaccatoF [sec]:")
print(len(audio)/float(44100))

#print(max(audio))
#print(min(audio))
"""
plot(audio[0*44100:int(len(audio)/float(44100))*44100])
plt.title("This is how StaccatoF looks like:")
plt.savefig('./StaccatoF.png')
"""
w = es.Windowing(type = 'hann')
spectrum = es.Spectrum()  # FFT() would return the complex FFT, here we just want the magnitude spectrum

pitch_values = []
pitch_confidence = []
frameSize = 2048
hopSize = 128
pitch_extractor = es.PitchYin(frameSize=2048, sampleRate=44100)
for frame in es.FrameGenerator(audio, frameSize=2048, hopSize=128, startFromZero=True):
    frequency, confidence = pitch_extractor(frame) # YINfft coge el spectrum como input OJO CUIDAO
    pitch_values.append(frequency)
    pitch_confidence.append(confidence)

print('len(pitch_values)')
print(len(pitch_values))
print(max(pitch_values))
print(min(pitch_values))
#print(pitch_values)
print('len(pitch_confidence)')
print(len(pitch_confidence))
print(max(pitch_confidence))
print(min(pitch_confidence))
#print(pitch_confidence)

# transpose to have it in a better shape
# we need to convert the list to an essentia.array first (== numpy.array of floats)
pitch_values = ES.array(pitch_values).T
pitch_confidence = ES.array(pitch_confidence).T
print('DESPUES')
print('len(pitch_values)')
print(len(pitch_values))
print(max(pitch_values))
print(min(pitch_values))
#print(pitch_values)
print('len(pitch_confidence)')
print(len(pitch_confidence))
print(max(pitch_confidence))
print(min(pitch_confidence))
#print(pitch_confidence)

# Pitch is estimated on frames. Compute frame time positions.
pitch_times = np.linspace(0.0,len(audio)/44100.0,len(pitch_values)) # está en segundos

# Plot the estimated pitch contour and confidence over time.
f, axarr = plt.subplots(2, sharex=True)
axarr[0].plot(pitch_times, pitch_values)
axarr[0].set_title('estimated pitch [Hz]')
axarr[1].plot(pitch_times, pitch_confidence)
axarr[1].set_title('pitch confidence')
plt.savefig('./pruebaYin.png')

"""
pitch_values = []
pitch_confidence = []
frameSize = 2048
hopSize = 128
pitch_extractor = es.PitchYinFFT(frameSize=2048, sampleRate=44100)

for fstart in range(0, len(audio)-frameSize, hopSize):
    frame = audio[fstart:fstart+frameSize]
    frequency, confidence = pitch_extractor(frame)
    pitch_values.append(frequency)
    pitch_confidence.append(confidence)

print('len(pitch_values)')
print(len(pitch_values))
print(pitch_values)
print(max(pitch_values))
print(min(pitch_values))
print('len(pitch_confidence)')
print(len(pitch_confidence))
print(pitch_confidence)
print(max(pitch_confidence))
print(min(pitch_confidence))
"""