"""
I want to use these 3 algorithms:

- PitchCREPE
- PitchYin
- PitchYinFFT

"""


# Plots
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (15, 6)

import numpy as np

import essentia.standard as es

audiofile = './pruebasAudio/turnaround120bpm.wav'

# Load audio file.
# 4 posibles loaders (3 normales, 1 caso especial)

loader = es.EqloudLoader(filename=audiofile, sampleRate=44100)
#loader = es.EasyLoader(filename=audiofile, sampleRate=16000)
#loader = es.MonoLoader(filename=audiofile, sampleRate=44100)
#loader = es.AudioLoader(filename=audiofile) # caso especial

audio = loader()
print("Duration of the audio sample [sec]:")
print(len(audio)/44100.0)

# Extract the pitch curve
"""
# 1) PitchCREPE (este no funciona, NO ENCUENTRA LA FUNCION DENTRO DE es.standard)
# ATENCIÓN: The required sample rate of input signal is 16 KHz (modificar arriba)
pitch_extractor = es.PitchCREPE(hopSize=128) #hopSize?
times, pitch_values, pitch_confidence, activations = pitch_extractor(audio)

# 2) PitchYin (este TAMPOCO funciona, parece que sí pero luego no carga el output)
pitch_extractor = es.PitchYin(frameSize=2048)
pitch_values, pitch_confidence = pitch_extractor(audio)

print(pitch_values)
"""
# 3) PitchYinFFT (el único que SÍ funciona)
pitch_extractor = es.PitchYinFFT(frameSize=2048) 
pitch_values, pitch_confidence = pitch_extractor(audio) # Me dan VALORES, no arrays :(
# VALE CREO QUE TENGO QUE HACER LO DE LOS FRAMES ANTES !!!!!

print(pitch_values)
print(pitch_confidence)

# Pitch is estimated on frames. Compute frame time positions.
pitch_times = np.linspace(0.0,len(audio)/44100.0,len(pitch_values) )

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
