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
print('---- 1. PitchYin')
print('---- 2. PitchYinFFT') # windowing, spectrum, etc.
print('---- 3. PitchCREPE') # caso especial (sampleRate=16000)
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
elif(loader_type == 2):
    str_loader = 'Easy'
    loader = es.EasyLoader(filename=audiofile, sampleRate=sampleRate)
elif(loader_type == 3):
    str_loader = 'Mono'
    loader = es.MonoLoader(filename=audiofile, sampleRate=sampleRate)
else:
    str_loader = 'Audio' # caso especial
    loader = es.AudioLoader(filename=audiofile)

path = './RESULTS/LEVEL'+str(level)+str_loader+str(sampleRate) # falta poner algoritmo
os.mkdir(path)
audio = loader()
print("Duration of the audio sample [sec]:")
print(len(audio)/float(sampleRate))

plot(audio[0*sampleRate:int(len(audio)/float(sampleRate))*sampleRate])
plt.title("This is how "+audiofiles[4]+" looks like:") # 4=i
plt.savefig(str(path)+'/'+audiofiles[4]+'.png') # 4=i

# create a hann window?
# los valores de audio entiendo que son intensidades or what (no pasan nunca del 1 y pueden ser negativas)
w = es.Windowing(type = 'hann')
frameSize=2048
hopSize=128
i=0
for fstart in range(0, len(audio)-frameSize, hopSize):
    i=i+1
    
    frame = audio[fstart:fstart+frameSize]
    print(frame)
    #mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
    #mfccs.append(mfcc_coeffs)
   # melbands.append(mfcc_bands)


"""
# frame-wise processing
mfccs = []
melbands = []
melbands_log = []

for frame in FrameGenerator(audio, frameSize=1024, hopSize=512, startFromZero=True):
    mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
    mfccs.append(mfcc_coeffs)
    melbands.append(mfcc_bands)
    melbands_log.append(logNorm(mfcc_bands))
    pitch_extractor = es.PitchYinFFT(frameSize=2048)
    pitch_values, pitch_confidence = pitch_extractor(audio)
"""

# hacer función para escoger uno de los 3 algoritmos estudiados (y sus características)
# Extract the pitch curve
"""
# 1) PitchCREPE (RuntimeError: In PitchCREPE.compute: TensorflowPredict: This algorithm 
is not configured. To configure this algorithm you should specify a valid `graphFilename` 
or `savedModel` as input parameter.)
# ATENCIÓN: The required sample rate of input signal is 16 KHz (modificar arriba)
pitch_extractor = es.PitchCREPE(hopSize=128) #hopSize?
times, pitch_values, pitch_confidence, activations = pitch_extractor(audio)

# 2) PitchYin (este TAMPOCO funciona, parece que sí pero luego no carga el output)
pitch_extractor = es.PitchYin(frameSize=2048)
pitch_values, pitch_confidence = pitch_extractor(audio)

print(pitch_values)

# 3) PitchYinFFT (el único que SÍ funciona)
pitch_extractor = es.PitchYinFFT(frameSize=2048, sampleRate=44100)
pitch_values, pitch_confidence = pitch_extractor(audio) # Me dan VALORES, no arrays :(
# VALE CREO QUE TENGO QUE HACER frame-wise processing??
# mirar apartado COMPUTATIONS ON FRAMES (de essentia python tutorial) 
# YA LO ESTOY PILLANDO!! (mirar apuntes)

# Hacer otra función que te plotee los resultados del algoritmo (pitch_values and pitch_confidence)
print(pitch_values)
print(pitch_confidence)

# Pitch is estimated on frames. Compute frame time positions.
pitch_times = np.linspace(0.0,len(audio)/44100.0,len(pitch_values))

# Plot the estimated pitch contour and confidence over time.
f, axarr = plt.subplots(2, sharex=True)
axarr[0].plot(pitch_times, pitch_values)
axarr[0].set_title('estimated pitch [Hz]')
axarr[1].plot(pitch_times, pitch_confidence)
axarr[1].set_title('pitch confidence')
#plt.show() 
#plt.savefig('./timeWasMelo160bpm_results.png')
#plt.savefig('./GoodbyePorkPieHat60bpm_results.png')
plt.savefig('./turnaround120bpm_results_pitchYinFFT.png')

"""
# hasta aquí ya tengo trabajo, cuando tenga esto sigo adelante con las métricas de 
# evaluación de la librería mir_eval, los archivos midi y tota la pesca.
