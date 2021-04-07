
#source http://github.com/wiseman/py-webrtcvad/blob/master/example.py
#edited for purposes of thesis

from librosa import load
import numpy as np
import collections
import contextlib
import sys
import webrtcvad
import Playback



def Float2Pcm(sig, dtype='int16'):
    """Convert floating point signal with a range from -1 to 1 to PCM.
    Any signal values outside the interval [-1.0, 1.0) are clipped.
    Source:https://github.com/mgeier/python-audio/blob/master/audio-files/utility.py
    No dithering is used.
    Note that there are different possibilities for scaling floating
    point numbers to PCM numbers, this function implements just one of
    them.  For an overview of alternatives see
    http://blog.bjornroche.com/2009/12/int-float-int-its-jungle-out-there.html
    Parameters
    ----------
    sig : array_like
        Input array, must have floating point type.
    dtype : data type, optional
        Desired (integer) data type.
    Returns
    -------
    numpy.ndarray
        Integer data, scaled and clipped to the range of the given
        *dtype*.
    See Also
    --------
    pcm2float, dtype
    """
    sig = np.asarray(sig)
    if sig.dtype.kind != 'f':
        raise TypeError("'sig' must be a float array")
    dtype = np.dtype(dtype)
    if dtype.kind not in 'iu':
        raise TypeError("'dtype' must be an integer type")

    i = np.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)


def Pcm2Float(sig, dtype='float64'):
    """Convert PCM signal to floating point with a range from -1 to 1.
    Use dtype='float32' for single precision.
    Source:https://github.com/mgeier/python-audio/blob/master/audio-files/utility.py
    Parameters
    ----------
    sig : array_like
        Input array, must have integral type.
    dtype : data type, optional
        Desired (floating point) data type.
    Returns
    -------
    numpy.ndarray
        Normalized floating point data.
    """
    sig = np.asarray(sig)
    if sig.dtype.kind not in 'iu':
        raise TypeError("'sig' must be an array of integers")
    dtype = np.dtype(dtype)
    if dtype.kind != 'f':
        raise TypeError("'dtype' must be a floating point type")

    i = np.iinfo(sig.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig.astype(dtype) - offset) / abs_max


def FrameGenerator(frame_duration_ms, audio, sample_rate):
	"""Generates audio frames from PCM audio data.
	Takes the desired frame duration in milliseconds, the PCM data, and
	the sample rate.
	Yields Frames of the requested duration.
	"""
	n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
	offset = 0
	timestamp = 0.0
	duration = (float(n) / sample_rate) / 2.0
	while offset + n < len(audio):
		yield Frame(audio[offset:offset + n], timestamp, duration)
		timestamp += duration
		offset += n


class Frame(object):
	"""Represents a "frame" of audio data."""
	def __init__(self, bytes, timestamp, duration):
		self.bytes = bytes
		self.timestamp = timestamp
		self.duration = duration


def SpeechInAudio(file, start=0.0, end=None):
	# need to find for audio file, because only extracted feature vector could be present
	file = Playback.GetAudioFile(file)	
	if end is not None:
		end = (end - start)*2 #duration
	audio, sample_rate = load(file, sr=8000, offset=start, duration=end)
	audio_pcm = Float2Pcm(audio)
	vad = webrtcvad.Vad(2)	# aggresiveness of vad
	frames = FrameGenerator(10, audio_pcm, sample_rate)
	voiced_frames = 0
	unvoiced_frames = 0
	for frame in frames:
		if vad.is_speech(frame.bytes, sample_rate):
			voiced_frames += 1
		else:
			unvoiced_frames += 1
	frames = list(frames)
	# at least 70% of sample is voiced
	if voiced_frames / (unvoiced_frames+voiced_frames) > 0.7:	
		return True		# Speech FOUND
	else:
		return False	# Speech NOT FOUND


if __name__ == "__main__":
	if len(sys.argv) == 2:
		# path to filename
		print("Speech in sample:", SpeechInAudio(sys.argv[1]))
		Playback.Playback(file=sys.argv[1])
	elif len(sys.argv) == 4:
		# path to filename, start in seconds, end in seconds
		print("Speech in sample:", SpeechInAudio(sys.argv[1], float(sys.argv[2]), float(sys.argv[3])))
		Playback.Playback(file=sys.argv[1], sim_list=[float(sys.argv[2]), float(sys.argv[3])-float(sys.argv[2])])	#start and end in seconds
