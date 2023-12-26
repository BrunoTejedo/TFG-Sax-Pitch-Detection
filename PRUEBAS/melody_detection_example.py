"""
In this tutorial we will analyse the pitch contour of the predominant melody in an audio 
recording using the PredominantPitchMelodia algorithm. This algorithm outputs a time series 
(sequence of values) with the instantaneous pitch value (in Hertz) of the perceived melody. 
It can be used with both monophonic and polyphonic signals.
"""


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

# For embedding audio player
import IPython

# Plots
import matplotlib.pyplot as plt
from pylab import plot, show, figure, imshow
plt.rcParams['figure.figsize'] = (15, 6)

import numpy

import essentia.standard as es

audiofile = './pruebasAudio/timeWasMelo160bpm.wav'

# Load audio file.
# It is recommended to apply equal-loudness filter for PredominantPitchMelodia.
loader = es.EqloudLoader(filename=audiofile, sampleRate=44100)
audio = loader()
print("Duration of the audio sample [sec]:")
print(len(audio)/44100.0)

# Extract the pitch curve
# PitchMelodia takes the entire audio signal as input (no frame-wise processing is required).

pitch_extractor = es.PredominantPitchMelodia(frameSize=2048, hopSize=128)
pitch_values, pitch_confidence = pitch_extractor(audio)

# Pitch is estimated on frames. Compute frame time positions.
pitch_times = numpy.linspace(0.0,len(audio)/44100.0,len(pitch_values) )

# Plot the estimated pitch contour and confidence over time.
f, axarr = plt.subplots(2, sharex=True)
axarr[0].plot(pitch_times, pitch_values)
axarr[0].set_title('estimated pitch [Hz]')
axarr[1].plot(pitch_times, pitch_confidence)
axarr[1].set_title('pitch confidence')
plt.show()

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
es.AudioWriter(filename=temp_dir.name + 'timeWasMelo160bpm_melody.mp3', format='wav')(es.StereoMuxer()(audio, synthesized_melody))

IPython.display.Audio(temp_dir.name + 'timeWasMelo160bpm_melody.mp3')


"""
NOTE SEGMENTATION AND CONVERTING TO MIDI

The PredominantPitchMelodia algorithm outputs pitch values in Hz, but we can also 
convert it to MIDI notes using the PitchContourSegmentation algorithm. Here is the 
default output it provides (tune the parameters for better note estimation).
"""
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

midi_file = temp_dir.name + '/extracted_melody.mid'
mid.save(midi_file)
print("MIDI file location:", midi_file)