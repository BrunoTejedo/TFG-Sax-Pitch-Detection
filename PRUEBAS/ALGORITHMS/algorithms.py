"""
Implementation of these 3 algorithms:

- PitchCREPE
- PitchYin
- PitchYinFFT

"""
import sys
import matplotlib.pyplot as plt
import essentia.standard as es
import utils

plt.rcParams['figure.figsize'] = (15, 6)

try:
    level = int(sys.argv[1])
except IndexError:
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

if algorithm == 3 and loader_type == 4:
    print("Error: PitchCREPE Algorithm and AudioLoader function are incompatible")
    sys.exit()

print('Select sampleRate (Hz):')
print('---- 16000, 32000, 44100, 48000, ...')
sample_rate = int(input())

if algorithm == 3 and sample_rate != 16000:
    print("Error: PitchCREPE Algorithm must use 16 kHz as sampleRate")
    sys.exit()

if loader_type == 4 and sample_rate != 44100:
    print("Error: AudioLoader only accepts 44100 Hz as sampleRate")
    sys.exit()

print('Select frameSize:')
print('---- 0, 1024, 2048, ...(for the CREPE algorithm it is not necessary, select 0)')
frame_size = int(input())

print('Select hopSize:')
print('---- 10, 128, 512, ... (for the CREPE algorithm 10 is recommended)')
hop_size = int(input())

audiofile_names = utils.list_audiofile_names(level)
audiofiles = utils.list_audiofiles(audiofile_names, level)

def process_audiofiles(loader_class, pitch_computation_func):
    path = utils.mkdir_results(level, str_algorithm, str_loader, sample_rate, frame_size, hop_size)
    if algorithm != 3:
        for j, i in enumerate(audiofiles):
            loader = loader_class(filename=i, sampleRate=sample_rate)
            audio = loader()
            print("Duration of the audio sample [sec]:", len(audio) / float(sample_rate))
            pitch_values, pitch_confidence, pitch_times = pitch_computation_func(audio)
            onsets, durations, notes = es.PitchContourSegmentation(hopSize=hop_size, sampleRate=sample_rate)(pitch_values, audio)
            utils.save_midi_csv(path, audiofile_names[j], notes, onsets, durations)
            utils.save_all_csv(path, audiofile_names[j], pitch_values, pitch_confidence, pitch_times)
            utils.plot_audio(audio, sample_rate, path, audiofile_names[j])
            utils.plot_estimated_pitch(pitch_times, pitch_values, pitch_confidence, path, audiofile_names[j])
    else:
        for j, i in enumerate(audiofiles):
            loader = loader_class(filename=i, sampleRate=sample_rate)
            audio = loader()
            print("Duration of the audio sample [sec]:", len(audio) / float(sample_rate))
            pitch_times, pitch_values, pitch_confidence, pitch_activations = pitch_computation_func(audio)
            onsets, durations, notes = es.PitchContourSegmentation(hopSize=hop_size, sampleRate=sample_rate)(pitch_values, audio)
            utils.save_midi_csv(path, audiofile_names[j], notes, onsets, durations)
            utils.save_all_csv(path, audiofile_names[j], pitch_values, pitch_confidence, pitch_times)
            utils.plot_audio(audio, sample_rate, path, audiofile_names[j])
            utils.plot_estimated_pitch(pitch_times, pitch_values, pitch_confidence, path, audiofile_names[j])

if algorithm == 1:
    str_algorithm = 'PitchYin'
    pitch_extractor = es.PitchYin(frameSize=frame_size, sampleRate=sample_rate)
    pitch_computation_func = lambda audio: utils.yin_computation(audio, frame_size, hop_size, pitch_extractor)

    if loader_type == 1:
        str_loader = 'EqLoud'
        process_audiofiles(es.EqloudLoader, pitch_computation_func)
    elif loader_type == 2:
        str_loader = 'Easy'
        process_audiofiles(es.EasyLoader, pitch_computation_func)
    elif loader_type == 3:
        str_loader = 'Mono'
        process_audiofiles(es.MonoLoader, pitch_computation_func)
    else:
        str_loader = 'Audio'
        sample_rate = 44100
        process_audiofiles(lambda filename, sampleRate: es.AudioLoader(filename=filename), pitch_computation_func)

elif algorithm == 2:
    str_algorithm = 'PitchYinFFT'
    pitch_extractor = es.PitchYinFFT(frameSize=frame_size, sampleRate=sample_rate)
    windowing = es.Windowing(type='hann')
    spectrum = es.Spectrum()
    pitch_computation_func = lambda audio: utils.yin_fft_computation(audio, frame_size, hop_size, pitch_extractor, windowing, spectrum)

    if loader_type == 1:
        str_loader = 'EqLoud'
        process_audiofiles(es.EqloudLoader, pitch_computation_func)
    elif loader_type == 2:
        str_loader = 'Easy'
        process_audiofiles(es.EasyLoader, pitch_computation_func)
    elif loader_type == 3:
        str_loader = 'Mono'
        process_audiofiles(es.MonoLoader, pitch_computation_func)
    else:
        str_loader = 'Audio'
        sample_rate = 44100
        process_audiofiles(lambda filename, sampleRate: es.AudioLoader(filename=filename), pitch_computation_func)

else:
    sample_rate = 16000
    frame_size = 0
    graph = utils.ask_about_crepe()
    str_algorithm, pitch_extractor = utils.decide_graph_crepe(graph, hop_size)

    if loader_type == 1:
        str_loader = 'EqLoud'
        process_audiofiles(es.EqloudLoader, lambda audio: pitch_extractor(audio))
    elif loader_type == 2:
        str_loader = 'Easy'
        process_audiofiles(es.EasyLoader, lambda audio: pitch_extractor(audio))
    elif loader_type == 3:
        str_loader = 'Mono'
        process_audiofiles(es.MonoLoader, lambda audio: pitch_extractor(audio))
    else:
        print("ERROR")
        sys.exit()
