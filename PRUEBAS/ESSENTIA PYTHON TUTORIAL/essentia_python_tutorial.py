
"""
ESSENTIA PYTHON TUTORIAL

https://essentia.upf.edu/essentia_python_tutorial.html

There are two modes of using Essentia, standard and streaming, and in this section, 
we will focus on the STANDARD mode. See next section for the streaming mode.

We will have a look at some basic functionality:

- how to load an audio

- how to perform some numerical operations such as FFT

- how to plot results

- how to output results to a file

"""

# first, we need to import our essentia module. It is aptly named 'essentia'!
import essentia

# there are two operating modes in essentia which (mostly) have the same algorithms
# they are accessible via two submodules:
import essentia.standard
import essentia.streaming

# let's have a look at what is in there
print(dir(essentia.standard))

# you can also do it by using autocompletion in Jupyter/IPython, typing "essentia.standard." and pressing Tab

"""
This list contains all Essentia algorithms available in standard mode. You can have an
inline help for the algorithms you are interested in using help command (you can also
see it by typing MFCC in Jupyter/IPython). You can also use our online algorithm reference.
"""

#help(essentia.standard.MFCC)

"""
INSTANTIATING OUR FIRST ALGORITHM, LOADING SOME AUDIO

Before you can use algorithms in Essentia, you first need to instantiate (create) them. 
When doing so, you can give them parameters which they may need to work properly, 
such as the filename of the audio file in the case of an audio loader.

Once you have instantiated an algorithm, nothing has happened yet, but your algorithm 
is ready to be used and works like a function, that is, you have to call it to make 
stuff happen (technically, it is a function object).

Essentia has a selection of audio loaders:

- AudioLoader: the most generic one, returns the audio samples, sampling rate 
                and number of channels, and some other related information

- MonoLoader: returns audio, down-mixed and resampled to a given sampling rate

- EasyLoader: a MonoLoader which can optionally trim start/end slices and 
                rescale according to a ReplayGain value

- EqloudLoader: an EasyLoader that applies an equal-loudness filtering to the audio
"""

# we start by instantiating the audio loader:
loader = essentia.standard.MonoLoader(filename='../pruebasAudio/GoodbyePorkPieHat60bpm.wav')

# and then we actually perform the loading:
audio = loader()

# This is how the audio we want to process sounds like
import IPython #esto no funciona desde la terminal wsl
IPython.display.Audio('../pruebasAudio/GoodbyePorkPieHat60bpm.wav')

# By default, the MonoLoader will output audio with 44100Hz sample rate downmixed 
# to mono. To make sure that this actually worked, let’s plot a 1-second slice 
# of audio, from t = 1 sec to t = 2 sec:

# pylab contains the plot() function, as well as figure, etc... (same names as Matlab)
from pylab import plot, show, figure, imshow
# %matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (15, 6) # set plot sizes to something larger than default

plot(audio[1*44100:2*44100])
plt.title("This is how the 2nd second of this audio looks like:")
#show()
plt.savefig('./Input_audio.png')


"""
COMPUTING SPECTRUM, MEL BANDS ENERGIES, AND MFCCs

So let’s say that we want to compute spectral energy in mel bands and the 
associated MFCCs for the frames in our audio.

We will need the following algorithms: Windowing, Spectrum, MFCC. 
For windowing, we’ll specify to use Hann window.
"""

from essentia.standard import *
w = Windowing(type = 'hann')
spectrum = Spectrum()  # FFT() would return the complex FFT, here we just want the magnitude spectrum
mfcc = MFCC()

"""
Once algorithms have been instantiated, they work like normal functions. 
Note that the MFCC algorithm returns two values: the band energies and the 
coefficients. Let’s compute and plot the spectrum, mel band energies, 
and MFCCs for a frame of audio:
"""

frame = audio[6*44100 : 6*44100 + 1024]
spec = spectrum(w(frame))
mfcc_bands, mfcc_coeffs = mfcc(spec)

plot(spec)
plt.title("The spectrum of a frame:")
#show()
plt.savefig('./spectrum.png')

plot(mfcc_bands)
plt.title("Mel band spectral energies of a frame:")
#show()
plt.savefig('./mel_bands.png')

plot(mfcc_coeffs)
plt.title("First 13 MFCCs of a frame:")
#show()
plt.savefig('./mfcc_coeffs.png')

"""
In the case of mel band energies, sometimes you may want to apply log normalization, 
which can be done using UnaryOperator. Using this algorithm we can do different 
types of normalization on vectors.
"""

logNorm = UnaryOperator(type='log')
plot(logNorm(mfcc_bands))
plt.title("Log-normalized mel band spectral energies of a frame:")
#show()
plt.savefig('./logNormMFCCbands')

"""
COMPUTATIONS ON FRAMES
Now let’s compute the mel band energies and MFCCs in all frames.

A typical Matlab-like way we would do it is by slicing the frames manually (the first 
frame starts at the moment 0, that is, with the first audio sample):
"""

mfccs = []
melbands = []
frameSize = 1024
hopSize = 512

for fstart in range(0, len(audio)-frameSize, hopSize):
    frame = audio[fstart:fstart+frameSize]
    mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
    mfccs.append(mfcc_coeffs)
    melbands.append(mfcc_bands)
    
"""
This is OK, but there is a much nicer way of computing frames in Essentia by using 
FrameGenerator, the FrameCutter algorithm wrapped into a Python generator:
"""

mfccs = []
melbands = []
melbands_log = []

for frame in FrameGenerator(audio, frameSize=1024, hopSize=512, startFromZero=True):
    mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
    mfccs.append(mfcc_coeffs)
    melbands.append(mfcc_bands)
    melbands_log.append(logNorm(mfcc_bands))

# transpose to have it in a better shape
# we need to convert the list to an essentia.array first (== numpy.array of floats)
mfccs = essentia.array(mfccs).T
melbands = essentia.array(melbands).T
melbands_log = essentia.array(melbands_log).T

# and plot
imshow(melbands[:,:], aspect = 'auto', origin='lower', interpolation='none')
plt.title("Mel band spectral energies in frames")
#show()
plt.savefig('./melBandINframes')

imshow(melbands_log[:,:], aspect = 'auto', origin='lower', interpolation='none')
plt.title("Log-normalized mel band spectral energies in frames")
#show()
plt.savefig('./logNormMFCCbandsINframes')

imshow(mfccs[1:,:], aspect='auto', origin='lower', interpolation='none')
plt.title("MFCCs in frames")
#show()
plt.savefig('./MFCCsINframes')

"""
You can configure frame and hop size of the frame generator, and whether to start the first 
frame or to center it at zero position in time. For the complete list of available parameters 
see the documentation for the FrameCutter.

Note, that when plotting MFCCs, we ignored the first coefficient to disregard the power 
of the signal and only plot its spectral shape.
"""

"""
STORING RESULTS TO POOL

A Pool is a container similar to a C++ map or Python dict which can contain any type of
values (easy in Python, not as much in C++…). Values are stored in there using a name
which represents the full path to these values with dot (.) characters used as separators. 
You can think of it as a directory tree, or as namespace(s) plus a local name.

Examples of valid names are: "bpm", "lowlevel.mfcc", "highlevel.genre.rock.probability", etc…

Let’s redo the previous computations using a pool. The pool has the nice advantage 
that the data you get out of it is already in an essentia.array format (which is equal 
to numpy.array of floats), so you can call transpose (.T) directly on it.
"""

pool = essentia.Pool()

for frame in FrameGenerator(audio, frameSize = 1024, hopSize = 512, startFromZero=True):
    mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
    pool.add('lowlevel.mfcc', mfcc_coeffs)
    pool.add('lowlevel.mfcc_bands', mfcc_bands)
    pool.add('lowlevel.mfcc_bands_log', logNorm(mfcc_bands))

imshow(pool['lowlevel.mfcc_bands'].T, aspect = 'auto', origin='lower', interpolation='none')
plt.title("Mel band spectral energies in frames")
#show()
plt.savefig('./melBandINframes2')

imshow(pool['lowlevel.mfcc_bands_log'].T, aspect = 'auto', origin='lower', interpolation='none')
plt.title("Log-normalized mel band spectral energies in frames")
#show()
plt.savefig('./logNormMFCCbandsINframes2')

imshow(pool['lowlevel.mfcc'].T[1:,:], aspect='auto', origin='lower', interpolation='none')
plt.title("MFCCs in frames")
#show()
plt.savefig('./MFCCsINframes2')

"""
AGGREGATION AND FILE OUTPUT

As we are using Python, we could use its facilities for writing data to a file, but 
for the sake of this tutorial let’s do it using the YamlOutput algorithm, which 
writes a pool in a file using the YAML or JSON format.
"""

output = YamlOutput(filename = 'mfcc.sig') # use "format = 'json'" for JSON output
output(pool)

# or as a one-liner:
YamlOutput(filename = 'mfcc.sig')(pool)

"""
This can take a while as we actually write the MFCCs for all the frames, which can be 
quite heavy depending on the duration of your audio file.

Now let’s assume we do not want all the frames but only the mean and standard deviation 
of those frames. We can do this using the PoolAggregator algorithm on our pool with frame 
value to get a new pool with the aggregated descriptors (check the documentation for this 
algorithm to get an idea of other statistics it can compute):
"""

# compute mean and variance of the frames
aggrPool = PoolAggregator(defaultStats = [ 'mean', 'stdev' ])(pool)

print('Original pool descriptor names:')
print(pool.descriptorNames())
print('')
print('Aggregated pool descriptor names:')
print(aggrPool.descriptorNames())
print('')

# and ouput those results in a file
YamlOutput(filename = 'mfccaggr.sig')(aggrPool)


"""
SUMMARY AND MORE EXAMPLES

There is not much more to know for using Essentia in standard mode in Python, the basics are:

- instantiate and configure algorithms

- use them to compute some results

- and that’s pretty much it!

You can find various Python examples in the src/examples/python folder in the source code, for example:

- computing spectral centroid (example_spectral_spectralcentroid.py)

- onset detection (example_rhythm_onsetdetection.py)

- predominant melody detection (example_pitch_predominantmelody.py and example_pitch_predominantmelody_by_steps.py)
"""
