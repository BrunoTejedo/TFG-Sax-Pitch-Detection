import librosa
import mido
import numpy as np
import string
import essentia.standard as es
import csv
import mir_eval.melody as mir
"""
# LegatoA
with open('./Audio/LEVEL1/CSV/LegatoA_midinotes.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
    
data_array = np.array(data)
data_array = np.array(data, dtype=float)
data_array = np.transpose(data_array)
ref_freq = data_array[0] # Hz

with open('./Audio/LEVEL1/CSV/LegatoA_midionsets.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
    
data_array = np.array(data)
data_array = np.array(data, dtype=float)
data_array = np.transpose(data_array)
ref_time = data_array[0] # seconds

with open('./RESULTS/LEVEL1PitchCREPEmediumMono_16000_0_10/times/LegatoA_times.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
    
data_array = np.array(data)
data_array = np.array(data, dtype=float)
data_array = np.transpose(data_array)
est_time = data_array[0] # seconds

with open('./RESULTS/LEVEL1PitchCREPEmediumMono_16000_0_10/values/LegatoA_values.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
    
data_array = np.array(data)
data_array = np.array(data, dtype=float)
data_array = np.transpose(data_array)
est_freq = data_array[0] # Hz

with open('./RESULTS/LEVEL1PitchCREPEmediumMono_16000_0_10/confidence/LegatoA_confidence.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
    
data_array = np.array(data)
data_array = np.array(data, dtype=float)
data_array = np.transpose(data_array)
est_voicing = data_array[0] # prob

print(ref_time)

ref_voicing, ref_cent, est_voicing, est_cent = mir.to_cent_voicing(ref_time, ref_freq, est_time, est_freq, est_voicing)
print(ref_voicing)
print(ref_cent)
print(est_voicing)
print(est_cent)

kdncdkc
"""
# LegatoA

with open('./RESULTS/LEVEL1PitchCREPEmediumMono_16000_0_10/times/LegatoA_times.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
    
data_array = np.array(data)
data_array = np.array(data, dtype=float)
data_array = np.transpose(data_array)
pitch_times = data_array[0] # seconds

with open('./RESULTS/LEVEL1PitchCREPEmediumMono_16000_0_10/values/LegatoA_values.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
    
data_array = np.array(data)
data_array = np.array(data, dtype=float)
data_array = np.transpose(data_array)
pitch_values = data_array[0] # Hz

with open('./RESULTS/LEVEL1PitchCREPEmediumMono_16000_0_10/confidence/LegatoA_confidence.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
    
data_array = np.array(data)
data_array = np.array(data, dtype=float)
data_array = np.transpose(data_array)
pitch_confidence = data_array[0] # prob

with open('./Audio/LEVEL1/CSV/LegatoA_midinotes.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
    
data_array = np.array(data)
data_array = np.array(data, dtype=float)
data_array = np.transpose(data_array)
notes = data_array[0] # Hz

with open('./Audio/LEVEL1/CSV/LegatoA_midionsets.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
    
data_array = np.array(data)
data_array = np.array(data, dtype=float)
data_array = np.transpose(data_array)
onsets = data_array[0] # seconds

with open('./Audio/LEVEL1/CSV/LegatoA_mididurations.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
    
data_array = np.array(data)
data_array = np.array(data, dtype=float)
data_array = np.transpose(data_array)
durations = data_array[0] # seconds

offsets = onsets + durations # finales de notas
silence_durations = list(onsets[1:] - offsets[:-1])
print(silence_durations)

# tengo que sacar ahora ref_times, ref_values, ref_confidence
ref_times = pitch_times
ref_values = np.zeros(len(ref_times))
print(ref_values)
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
    
print(ref_values)

ref_voicing, ref_cent, est_voicing, est_cent = mir.to_cent_voicing(ref_times, ref_values, pitch_times, pitch_values, pitch_confidence)

raw_pitch = mir.raw_pitch_accuracy(ref_voicing, ref_cent, est_voicing, est_cent, cent_tolerance=50)
print(raw_pitch)

overall_accuracy = mir.overall_accuracy(ref_voicing, ref_cent, est_voicing, est_cent, cent_tolerance=50)
print(overall_accuracy)

mir.validate_voicing(ref_voicing, est_voicing)
print(mir.validate_voicing(ref_voicing, est_voicing))
mir.validate(ref_voicing, ref_cent, est_voicing, est_cent)
print(mir.validate(ref_voicing, ref_cent, est_voicing, est_cent))
"""
print(ref_voicing)
print(ref_cent)
print(est_voicing)
print(est_cent)
"""