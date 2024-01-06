"""
In this tutorial we will analyse the pitch contour of the predominant melody in an audio 
recording using the PredominantPitchMelodia algorithm. This algorithm outputs a time series 
(sequence of values) with the instantaneous pitch value (in Hertz) of the perceived melody. 
It can be used with both monophonic and polyphonic signals.
"""

# For embedding audio player
import IPython

# Plots
import matplotlib.pyplot as plt
from pylab import plot, show, figure, imshow
plt.rcParams['figure.figsize'] = (15, 6)

import numpy

import essentia.standard as es

#audiofile = './pruebasAudio/timeWasMelo160bpm.wav'
#audiofile = './pruebasAudio/GoodbyePorkPieHat60bpm.wav'
audiofile = './pruebasAudio/StaccatoF.wav'

# Load audio file.
# It is recommended to apply equal-loudness filter for PredominantPitchMelodia.
loader = es.EqloudLoader(filename=audiofile, sampleRate=44100)
audio = loader()
print("Duration of the audio sample [sec]:")
print(len(audio)/44100.0) #61.1824716553288

# Extract the pitch curve
# PitchMelodia takes the entire audio signal as input (no frame-wise processing is required).

# frameSize? hopSize?
pitch_extractor = es.PredominantPitchMelodia(frameSize=2048, hopSize=128)
pitch_values, pitch_confidence = pitch_extractor(audio)
# este es el resultado del algoritmo, (Hz,prob)
print('frameSize = 2048')
print('hopSize = 128')
print('len(pitch_values)')
print(len(pitch_values)) #21081, por alguna razón es este valor 
print(max(pitch_values))
print(min(pitch_values))
print('len(pitch_confidence)')
print(len(pitch_confidence))
print(max(pitch_confidence))
print(min(pitch_confidence))
"""
print(pitch_values[2000]) #698.4669
print(pitch_confidence[2000]) #5.8742266e-05
print(max(pitch_confidence)) #0.15614718
print(min(pitch_confidence)) #0.0
"""

# Pitch is estimated on frames. Compute frame time positions.
pitch_times = numpy.linspace(0.0,len(audio)/44100.0,len(pitch_values) )
print('len(pitch_times)')
print(len(pitch_times))
#[0.00000000e+00 2.90239429e-03 5.80478858e-03 ... 6.11766669e+01 6.11795693e+01 6.11824717e+01]
# print(len(pitch_times)) #21081 = 61.1824716553288/0.002902394291
# pitch_times es un vector que divide el tiempo del audio en PARTES IGUALES (frames),
# lo que NO SÉ EN CUANTAS (quizás hop n frame size has something to do)
###### NO: depende de len(pitch_values), que esto ya no sé cómo va

# Plot the estimated pitch contour and confidence over time.
f, axarr = plt.subplots(2, sharex=True)
axarr[0].plot(pitch_times, pitch_values)
axarr[0].set_title('estimated pitch [Hz]')
axarr[1].plot(pitch_times, pitch_confidence)
axarr[1].set_title('pitch confidence')
#plt.show() 
#plt.savefig('./timeWasMelo160bpm_results.png')
#plt.savefig('./GoodbyePorkPieHat60bpm_results.png')
plt.savefig('./StaccatoF_results.png')

"""
The zero pitch value correspond to unvoiced audio segments with a very low pitch 
confidence according to the algorithm’s estimation. You can force estimations on 
those as well by setting the guessUnvoiced parameter.
"""

"""
Let’s listen to the estimated pitch and compare it to the original audio. 
To this end we will generate a sine wave signal following the estimated pitch, 
using the mir_eval Python package (make sure to install it with pip install 
                                   mir_eval to be able to run this code).
"""
IPython.display.Audio(audiofile)

from mir_eval.sonify import pitch_contour

from tempfile import TemporaryDirectory
temp_dir = TemporaryDirectory()

# Essentia operates with float32 ndarrays instead of float64, so let's cast it.
synthesized_melody = pitch_contour(pitch_times, pitch_values, 44100).astype(numpy.float32)[:len(audio)]
#es.AudioWriter(filename='./timeWasMelo160bpm_melody.wav', format='wav')(es.StereoMuxer()(audio, synthesized_melody))
#es.AudioWriter(filename='./GoodbyePorkPieHat60bpm_melody.wav', format='wav')(es.StereoMuxer()(audio, synthesized_melody))
es.AudioWriter(filename='./StaccatoF_melody.wav', format='wav')(es.StereoMuxer()(audio, synthesized_melody))
#temp_dir.name + 
#IPython.display.Audio('./timeWasMelo160bpm_melody.wav') 
#IPython.display.Audio('./GoodbyePorkPieHat60bpm_melody.wav')
IPython.display.Audio('./turnaround120bpm_melody.wav')
# guay, pero porque me lo guarda en una carpeta random???


"""
NOTE SEGMENTATION AND CONVERTING TO MIDI

The PredominantPitchMelodia algorithm outputs pitch values in Hz, but we can also 
convert it to MIDI notes using the PitchContourSegmentation algorithm. Here is the 
default output it provides (tune the parameters for better note estimation).
"""
#hoSize again????
onsets, durations, notes = es.PitchContourSegmentation(hopSize=128)(pitch_values, audio)
print("MIDI notes:", notes) # Midi pitch number
print("MIDI note onsets:", onsets)
print("MIDI note durations:", durations)

"""
We can now export results to a MIDI file. We will use mido Python package 
(which you can install with pip install mido) to do generate the .mid file. 
You can test the result using the generated .mid file in a DAW.
"""

import mido

# PPQ BPM??? 96 n 120
PPQ = 96 # Pulses per quarter note.
BPM = 120 # Assuming a default tempo in Ableton to build a MIDI clip.
tempo = mido.bpm2tempo(BPM) # Microseconds per beat.

# Compute onsets and offsets for all MIDI notes in ticks.
# Relative tick positions start from time 0.
offsets = onsets + durations
silence_durations = list(onsets[1:] - offsets[:-1]) + [0]

mid = mido.MidiFile()
track = mido.MidiTrack()
mid.tracks.append(track)

for note, onset, duration, silence_duration in zip(list(notes), list(onsets), list(durations), silence_durations):
    track.append(mido.Message('note_on', note=int(note), velocity=64,
                              time=int(mido.second2tick(duration, PPQ, tempo))))
    track.append(mido.Message('note_off', note=int(note),
                              time=int(mido.second2tick(silence_duration, PPQ, tempo))))
#temp_dir.name + 
#midi_file = './timeWasMelo160bpm'+str(PPQ)+str(BPM)+'extracted_melody.mid'
#midi_file = './GoodbyePorkPieHat60bpm'+str(PPQ)+str(BPM)+'extracted_melody.mid'
midi_file = './StaccatoF'+str(PPQ)+str(BPM)+'extracted_melody.mid'
mid.save(midi_file)
print("MIDI file location:", midi_file)