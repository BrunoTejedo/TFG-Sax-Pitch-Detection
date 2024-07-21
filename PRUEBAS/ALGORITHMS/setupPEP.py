"""
GROUNDTRUTH INFORMATION:
    1st file to run

"""
import mido
import os
import numpy as np
import librosa


def extract_midi_info(file_path, output_folder):
    print(f'Extracting MIDI notes of {file_path} ...')
    mid = mido.MidiFile(file_path)
    notes = []
    onsets = []  # ticks
    durations = []  # ticks
    accumulated_time = 0

    for i in range(0, len(mid.tracks[0]) - 2):
        note = mid.tracks[0][i].note
        time = mid.tracks[0][i].time
        accumulated_time += time

        if i % 2 == 0:
            onsets.append(accumulated_time)
            notes.append(note)
        else:
            durations.append(time)

    notes = librosa.midi_to_hz(notes)  # Midi notes to Hz
    tempo = mido.bpm2tempo(120)  # we assume that BPM=120
    onsets = np.array(onsets)
    durations = np.array(durations)
    onsets = mido.tick2second(onsets, mid.ticks_per_beat, tempo)
    durations = mido.tick2second(durations, mid.ticks_per_beat, tempo)

    file_prefix = os.path.splitext(os.path.basename(file_path))[0]
    np.savetxt(f'{output_folder}/{file_prefix}_midinotes.csv', notes)
    np.savetxt(f'{output_folder}/{file_prefix}_midionsets.csv', onsets)
    np.savetxt(f'{output_folder}/{file_prefix}_mididurations.csv', durations)


def process_levels(levels):
    for level in levels:
        mid_folder = f'./Audio/{level}/MID'
        csv_folder = f'./Audio/{level}/CSV'
        os.makedirs(csv_folder, exist_ok=True)

        mid_files = os.listdir(mid_folder)
        for mid_file in mid_files:
            file_path = os.path.join(mid_folder, mid_file)
            extract_midi_info(file_path, csv_folder)


if __name__ == "__main__":
    LEVELS = ['LEVEL1', 'LEVEL2', 'LEVEL3']
    process_levels(LEVELS)
