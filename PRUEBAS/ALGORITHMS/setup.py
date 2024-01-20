# GROUNDTRUTH info
# 1st file to run

import mido
import os
import numpy as np
import librosa

for file in os.listdir('./Audio/LEVEL1/MID'):
    print('Extracting MIDI notes of ' + file + ' ...')
    mid = mido.MidiFile('./Audio/LEVEL1/MID/'+file)
    notes = []
    onsets = [] # ticks
    durations = [] # ticks
    accumulated_time = 0
    for i in range(0,len(mid.tracks[0])-2):
        note = mid.tracks[0][i].note
        time = mid.tracks[0][i].time
        accumulated_time = accumulated_time + time
        if (i % 2) == 0:
            onsets.append(accumulated_time)
            notes.append(note)
        else:
            durations.append(time)
    notes = librosa.midi_to_hz(notes) # Midi notes to Hz
    tempo = mido.bpm2tempo(120) # we assume that BPM=120
    onsets = np.array(onsets)
    durations = np.array(durations)
    onsets = mido.tick2second(onsets,mid.ticks_per_beat,tempo)
    durations = mido.tick2second(durations,mid.ticks_per_beat,tempo)
    np.savetxt('./Audio/LEVEL1/CSV/'+file[:-4]+'_midinotes.csv', notes)
    np.savetxt('./Audio/LEVEL1/CSV/'+file[:-4]+'_midionsets.csv', onsets)
    np.savetxt('./Audio/LEVEL1/CSV/'+file[:-4]+'_mididurations.csv', durations)

for file in os.listdir('./Audio/LEVEL2/MID'):
    print('Extracting MIDI notes of ' + file + ' ...')
    mid = mido.MidiFile('./Audio/LEVEL2/MID/'+file)
    notes = []
    onsets = [] # ticks
    durations = [] # ticks
    accumulated_time = 0
    for i in range(0,len(mid.tracks[0])-2):
        note = mid.tracks[0][i].note
        time = mid.tracks[0][i].time
        accumulated_time = accumulated_time + time
        if (i % 2) == 0:
            onsets.append(accumulated_time)
            notes.append(note)
        else:
            durations.append(time)
    notes = librosa.midi_to_hz(notes) # Midi notes to Hz
    tempo = mido.bpm2tempo(120) # we assume that BPM=120
    onsets = np.array(onsets)
    durations = np.array(durations)
    onsets = mido.tick2second(onsets,mid.ticks_per_beat,tempo)
    durations = mido.tick2second(durations,mid.ticks_per_beat,tempo)
    durations = mido.tick2second(durations,mid.ticks_per_beat,tempo)
    np.savetxt('./Audio/LEVEL2/CSV/'+file[:-4]+'_midinotes.csv', notes)
    np.savetxt('./Audio/LEVEL2/CSV/'+file[:-4]+'_midionsets.csv', onsets)
    np.savetxt('./Audio/LEVEL2/CSV/'+file[:-4]+'_mididurations.csv', durations)

for file in os.listdir('./Audio/LEVEL3/MID'):
    print('Extracting MIDI notes of ' + file + ' ...')
    mid = mido.MidiFile('./Audio/LEVEL3/MID/'+file)
    notes = []
    onsets = [] # ticks
    durations = [] # ticks
    accumulated_time = 0
    for i in range(0,len(mid.tracks[0])-2):
        note = mid.tracks[0][i].note
        time = mid.tracks[0][i].time
        accumulated_time = accumulated_time + time
        if (i % 2) == 0:
            onsets.append(accumulated_time)
            notes.append(note)
        else:
            durations.append(time)
    notes = librosa.midi_to_hz(notes) # Midi notes to Hz
    tempo = mido.bpm2tempo(120) # we assume that BPM=120
    onsets = np.array(onsets)
    durations = np.array(durations)
    onsets = mido.tick2second(onsets,mid.ticks_per_beat,tempo)
    durations = mido.tick2second(durations,mid.ticks_per_beat,tempo)
    np.savetxt('./Audio/LEVEL3/CSV/'+file[:-4]+'_midinotes.csv', notes)
    np.savetxt('./Audio/LEVEL3/CSV/'+file[:-4]+'_midionsets.csv', onsets)
    np.savetxt('./Audio/LEVEL3/CSV/'+file[:-4]+'_mididurations.csv', durations)
    