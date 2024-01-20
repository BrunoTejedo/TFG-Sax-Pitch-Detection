"""
Implementation of these 3 algorithms:

- PitchCREPE
- PitchYin
- PitchYinFFT

"""
import utils
import essentia.standard as es
import sys
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (15, 6)

try:
    level = int(sys.argv[1])
except:
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
if algorithm==3 and loader_type==4:
    print("Error: PitchCREPE Algorithm and AudioLoader function are incompatible")
    sys.exit()

print('Select sampleRate (Hz):')
print('---- 16000, 32000, 44100, 48000, ...')
sampleRate = int(input())
if algorithm==3 and sampleRate!=16000:
    print("Error: PitchCREPE Algorithm must use 16 kHz as sampleRate")
    sys.exit()
if loader_type==4 and sampleRate!=44100:
    print("Error: AudioLoader only accepts 44100 Hz as sampleRate")
    sys.exit()
    
print('Select frameSize:')
print('---- 0, 1024, 2048, ...(for the CREPE algorithm it is not necessary, select 0)')
frameSize = int(input())

print('Select hopSize:')
print('---- 10, 128, 512, ... (for the CREPE algorithm 10 is recommended)')
hopSize = int(input())

audiofile_names = utils.list_audiofile_names(level)
audiofiles = utils.list_audiofiles(audiofile_names, level)

if (algorithm == 1):
    str_algorithm = 'PitchYin'
    pitch_extractor = es.PitchYin(frameSize=frameSize, sampleRate=sampleRate)
    if (loader_type == 1):
        str_loader = 'EqLoud'
        path = utils.mkdirResults(level, str_algorithm, str_loader, sampleRate, frameSize, hopSize)
        j=0
        for i in audiofiles:
            loader = es.EqloudLoader(filename=i, sampleRate=sampleRate)
            audio = loader()
            print("Duration of the audio sample [sec]:")
            print(len(audio)/float(sampleRate))
            pitch_values, pitch_confidence, pitch_times = utils.YinComputation(audio, frameSize, hopSize, pitch_extractor)
            onsets, durations, notes = es.PitchContourSegmentation(hopSize=128)(pitch_values, audio)
            utils.save_midi_csv(path, audiofile_names[j], notes, onsets, durations)
            utils.save_all_csv(path, audiofile_names[j], pitch_values, pitch_confidence, pitch_times)
            utils.plotAudio(audio, sampleRate, path, audiofile_names[j])
            utils.plot_estimated_pitch(pitch_times, pitch_values, pitch_confidence, path, audiofile_names[j])
            j=j+1
    elif (loader_type == 2):
        str_loader = 'Easy'
        path = utils.mkdirResults(level, str_algorithm, str_loader, sampleRate, frameSize, hopSize)
        j=0
        for i in audiofiles:
            loader = es.EasyLoader(filename=i, sampleRate=sampleRate)
            audio = loader()
            print("Duration of the audio sample [sec]:")
            print(len(audio)/float(sampleRate))
            pitch_values, pitch_confidence, pitch_times = utils.YinComputation(audio, frameSize, hopSize, pitch_extractor)
            onsets, durations, notes = es.PitchContourSegmentation(hopSize=128)(pitch_values, audio)
            utils.save_midi_csv(path, audiofile_names[j], notes, onsets, durations)
            utils.save_all_csv(path, audiofile_names[j], pitch_values, pitch_confidence, pitch_times)
            utils.plotAudio(audio, sampleRate, path, audiofile_names[j])
            utils.plot_estimated_pitch(pitch_times, pitch_values, pitch_confidence, path, audiofile_names[j])
            j=j+1
    elif (loader_type == 3):
        str_loader = 'Mono'
        path = utils.mkdirResults(level, str_algorithm, str_loader, sampleRate, frameSize, hopSize)
        j=0
        for i in audiofiles:
            loader = es.MonoLoader(filename=i, sampleRate=sampleRate)
            audio = loader()
            print("Duration of the audio sample [sec]:")
            print(len(audio)/float(sampleRate))
            pitch_values, pitch_confidence, pitch_times = utils.YinComputation(audio, frameSize, hopSize, pitch_extractor)
            onsets, durations, notes = es.PitchContourSegmentation(hopSize=128)(pitch_values, audio)
            utils.save_midi_csv(path, audiofile_names[j], notes, onsets, durations)
            utils.save_all_csv(path, audiofile_names[j], pitch_values, pitch_confidence, pitch_times)
            utils.plotAudio(audio, sampleRate, path, audiofile_names[j])
            utils.plot_estimated_pitch(pitch_times, pitch_values, pitch_confidence, path, audiofile_names[j])
            j=j+1
    else:
        str_loader = 'Audio'
        sampleRate = 44100
        path = utils.mkdirResults(level, str_algorithm, str_loader, sampleRate, frameSize, hopSize)
        j=0
        for i in audiofiles:
            loader = es.AudioLoader(filename=i)
            audio_vector = loader()
            audio = audio_vector[0]
            print("Duration of the audio sample [sec]:")
            print(len(audio)/float(sampleRate))
            pitch_values, pitch_confidence, pitch_times = utils.YinComputation(audio, frameSize, hopSize, pitch_extractor)
            onsets, durations, notes = es.PitchContourSegmentation(hopSize=128)(pitch_values, audio)
            utils.save_midi_csv(path, audiofile_names[j], notes, onsets, durations)
            utils.save_all_csv(path, audiofile_names[j], pitch_values, pitch_confidence, pitch_times)
            utils.plotAudio(audio, sampleRate, path, audiofile_names[j])
            utils.plot_estimated_pitch(pitch_times, pitch_values, pitch_confidence, path, audiofile_names[j])
            j=j+1
    
    
elif (algorithm == 2):
    str_algorithm = 'PitchYinFFT'
    pitch_extractor = es.PitchYinFFT(frameSize=frameSize, sampleRate=sampleRate)
    w = es.Windowing(type = 'hann')
    spectrum = es.Spectrum()
    if (loader_type == 1):
        str_loader = 'EqLoud'
        path = utils.mkdirResults(level, str_algorithm, str_loader, sampleRate, frameSize, hopSize)
        j=0
        for i in audiofiles:
            loader = es.EqloudLoader(filename=i, sampleRate=sampleRate)
            audio = loader()
            print("Duration of the audio sample [sec]:")
            print(len(audio)/float(sampleRate))
            pitch_values, pitch_confidence, pitch_times = utils.YinFFTComputation(audio, frameSize, hopSize, pitch_extractor, w, spectrum)
            onsets, durations, notes = es.PitchContourSegmentation(hopSize=128)(pitch_values, audio)
            utils.save_midi_csv(path, audiofile_names[j], notes, onsets, durations)
            utils.save_all_csv(path, audiofile_names[j], pitch_values, pitch_confidence, pitch_times)
            utils.plotAudio(audio, sampleRate, path, audiofile_names[j])
            utils.plot_estimated_pitch(pitch_times, pitch_values, pitch_confidence, path, audiofile_names[j])
            j=j+1
    elif (loader_type == 2):
        str_loader = 'Easy'
        path = utils.mkdirResults(level, str_algorithm, str_loader, sampleRate, frameSize, hopSize)
        j=0
        for i in audiofiles:
            loader = es.EasyLoader(filename=i, sampleRate=sampleRate)
            audio = loader()
            print("Duration of the audio sample [sec]:")
            print(len(audio)/float(sampleRate))
            pitch_values, pitch_confidence, pitch_times = utils.YinFFTComputation(audio, frameSize, hopSize, pitch_extractor, w, spectrum)
            onsets, durations, notes = es.PitchContourSegmentation(hopSize=128)(pitch_values, audio)
            utils.save_midi_csv(path, audiofile_names[j], notes, onsets, durations)
            utils.save_all_csv(path, audiofile_names[j], pitch_values, pitch_confidence, pitch_times)
            utils.plotAudio(audio, sampleRate, path, audiofile_names[j])
            utils.plot_estimated_pitch(pitch_times, pitch_values, pitch_confidence, path, audiofile_names[j])
            j=j+1
    elif (loader_type == 3):
        str_loader = 'Mono'
        path = utils.mkdirResults(level, str_algorithm, str_loader, sampleRate, frameSize, hopSize)
        j=0
        for i in audiofiles:
            loader = es.MonoLoader(filename=i, sampleRate=sampleRate)
            audio = loader()
            print("Duration of the audio sample [sec]:")
            print(len(audio)/float(sampleRate))
            pitch_values, pitch_confidence, pitch_times = utils.YinFFTComputation(audio, frameSize, hopSize, pitch_extractor, w, spectrum)
            onsets, durations, notes = es.PitchContourSegmentation(hopSize=128)(pitch_values, audio)
            utils.save_midi_csv(path, audiofile_names[j], notes, onsets, durations)
            utils.save_all_csv(path, audiofile_names[j], pitch_values, pitch_confidence, pitch_times)
            utils.plotAudio(audio, sampleRate, path, audiofile_names[j])
            utils.plot_estimated_pitch(pitch_times, pitch_values, pitch_confidence, path, audiofile_names[j])
            j=j+1
    else:
        str_loader = 'Audio'
        sampleRate = 44100
        path = utils.mkdirResults(level, str_algorithm, str_loader, sampleRate, frameSize, hopSize)
        j=0
        for i in audiofiles:
            loader = es.AudioLoader(filename=i)
            audio_vector = loader()
            audio = audio_vector[0]
            print("Duration of the audio sample [sec]:")
            print(len(audio)/float(sampleRate))
            pitch_values, pitch_confidence, pitch_times = utils.YinFFTComputation(audio, frameSize, hopSize, pitch_extractor, w, spectrum)
            onsets, durations, notes = es.PitchContourSegmentation(hopSize=128)(pitch_values, audio)
            utils.save_midi_csv(path, audiofile_names[j], notes, onsets, durations)
            utils.save_all_csv(path, audiofile_names[j], pitch_values, pitch_confidence, pitch_times)
            utils.plotAudio(audio, sampleRate, path, audiofile_names[j])
            utils.plot_estimated_pitch(pitch_times, pitch_values, pitch_confidence, path, audiofile_names[j])
            j=j+1
    
else:
    sampleRate = 16000
    frameSize = 0
    graph = utils.askAboutCREPE()
    str_algorithm, pitch_extractor = utils.decideGraphCREPE(graph, hopSize)
    if (loader_type == 1):
        str_loader = 'EqLoud'
        path = utils.mkdirResults(level, str_algorithm, str_loader, sampleRate, frameSize, hopSize)
        j=0
        for i in audiofiles:
            loader = es.EqloudLoader(filename=i, sampleRate=sampleRate)
            audio = loader()
            print("Duration of the audio sample [sec]:")
            print(len(audio)/float(sampleRate))
            pitch_times, pitch_values, pitch_confidence, pitch_activations = pitch_extractor(audio)
            onsets, durations, notes = es.PitchContourSegmentation(hopSize=128)(pitch_values, audio)
            utils.save_midi_csv(path, audiofile_names[j], notes, onsets, durations)
            utils.save_all_csv(path, audiofile_names[j], pitch_values, pitch_confidence, pitch_times)
            utils.plotAudio(audio, sampleRate, path, audiofile_names[j])
            utils.plot_estimated_pitch(pitch_times, pitch_values, pitch_confidence, path, audiofile_names[j])
            j=j+1
    elif (loader_type == 2):
        str_loader = 'Easy'
        path = utils.mkdirResults(level, str_algorithm, str_loader, sampleRate, frameSize, hopSize)
        j=0
        for i in audiofiles:
            loader = es.EasyLoader(filename=i, sampleRate=sampleRate)
            audio = loader()
            print("Duration of the audio sample [sec]:")
            print(len(audio)/float(sampleRate))
            pitch_times, pitch_values, pitch_confidence, pitch_activations = pitch_extractor(audio)
            onsets, durations, notes = es.PitchContourSegmentation(hopSize=128)(pitch_values, audio)
            utils.save_midi_csv(path, audiofile_names[j], notes, onsets, durations)
            utils.save_all_csv(path, audiofile_names[j], pitch_values, pitch_confidence, pitch_times)
            utils.plotAudio(audio, sampleRate, path, audiofile_names[j])
            utils.plot_estimated_pitch(pitch_times, pitch_values, pitch_confidence, path, audiofile_names[j])
            j=j+1
    elif (loader_type == 3):
        str_loader = 'Mono'
        path = utils.mkdirResults(level, str_algorithm, str_loader, sampleRate, frameSize, hopSize)
        j=0
        for i in audiofiles:
            loader = es.MonoLoader(filename=i, sampleRate=sampleRate)
            audio = loader()
            print("Duration of the audio sample [sec]:")
            print(len(audio)/float(sampleRate))
            pitch_times, pitch_values, pitch_confidence, pitch_activations = pitch_extractor(audio)
            onsets, durations, notes = es.PitchContourSegmentation(hopSize=128)(pitch_values, audio)
            utils.save_midi_csv(path, audiofile_names[j], notes, onsets, durations)
            utils.save_all_csv(path, audiofile_names[j], pitch_values, pitch_confidence, pitch_times)
            utils.plotAudio(audio, sampleRate, path, audiofile_names[j])
            utils.plot_estimated_pitch(pitch_times, pitch_values, pitch_confidence, path, audiofile_names[j])
            j=j+1
    else:
        print("ERROR")
        sys.exit()

