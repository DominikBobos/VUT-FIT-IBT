
#edited from http://github.com/wiseman/py-webrtcvad/blob/master/example.py

from librosa import load
import numpy as np
import collections
import contextlib
import sys
import webrtcvad
import Playback




# def Float2Pcm(audio):
# 	ints = (audio * 32767).astype(np.int16)
# 	little_endian = ints.astype('<u2')
# 	buff = little_endian.tobytes()
# 	return buff


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


def VadCollector(sample_rate, frame_duration_ms,
				  padding_duration_ms, vad, frames):
	"""Filters out non-voiced audio frames.
	Given a webrtcvad.Vad and a source of audio frames, yields only
	the voiced audio.
	Uses a padded, sliding window algorithm over the audio frames.
	When more than 90% of the frames in the window are voiced (as
	reported by the VAD), the collector triggers and begins yielding
	audio frames. Then the collector waits until 90% of the frames in
	the window are unvoiced to detrigger.
	The window is padded at the front and back to provide a small
	amount of silence or the beginnings/endings of speech around the
	voiced frames.
	Arguments:
	sample_rate - The audio sample rate, in Hz.
	frame_duration_ms - The frame duration in milliseconds.
	padding_duration_ms - The amount to pad the window, in milliseconds.
	vad - An instance of webrtcvad.Vad.
	frames - a source of audio frames (sequence or generator).
	Returns: A generator that yields PCM audio data.
	"""
	num_padding_frames = int(padding_duration_ms / frame_duration_ms)
	# We use a deque for our sliding window/ring buffer.
	ring_buffer = collections.deque(maxlen=num_padding_frames)
	# We have two states: TRIGGERED and NOTTRIGGERED. We start in the
	# NOTTRIGGERED state.
	triggered = False

	voiced_frames = []
	for frame in frames:
		is_speech = vad.is_speech(frame.bytes, sample_rate)

		sys.stdout.write('1' if is_speech else '0')
		if not triggered:
			ring_buffer.append((frame, is_speech))
			num_voiced = len([f for f, speech in ring_buffer if speech])
			# If we're NOTTRIGGERED and more than 90% of the frames in
			# the ring buffer are voiced frames, then enter the
			# TRIGGERED state.
			if num_voiced > 0.9 * ring_buffer.maxlen:
				triggered = True
				sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
				# We want to yield all the audio we see from now until
				# we are NOTTRIGGERED, but we have to start with the
				# audio that's already in the ring buffer.
				for f, s in ring_buffer:
					voiced_frames.append(f)
				ring_buffer.clear()
		else:
			# We're in the TRIGGERED state, so collect the audio data
			# and add it to the ring buffer.
			voiced_frames.append(frame)
			ring_buffer.append((frame, is_speech))
			num_unvoiced = len([f for f, speech in ring_buffer if not speech])
			# If more than 90% of the frames in the ring buffer are
			# unvoiced, then enter NOTTRIGGERED and yield whatever
			# audio we've collected.
			if num_unvoiced > 0.9 * ring_buffer.maxlen:
				sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
				triggered = False
				yield b''.join([f.bytes for f in voiced_frames])
				ring_buffer.clear()
				voiced_frames = []
	if triggered:
		sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
	sys.stdout.write('\n')
	# If we have any leftover voiced audio when we run out of input,
	# yield it.
	if voiced_frames:
		yield b''.join([f.bytes for f in voiced_frames])



class Frame(object):
	"""Represents a "frame" of audio data."""
	def __init__(self, bytes, timestamp, duration):
		self.bytes = bytes
		self.timestamp = timestamp
		self.duration = duration


def SpeechInAudio(file, start=0.0, end=None):
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
		print(frame.timestamp)
		if vad.is_speech(frame.bytes, sample_rate):
			voiced_frames += 1
		else:
			unvoiced_frames += 1
		# print(frame.bytes, frame.timestamp, frame.duration)
	frames = list(frames)
	# segments = VadCollector(sample_rate, 10, 300, vad, frames)
	# print(list(segments))

	if voiced_frames / (unvoiced_frames+voiced_frames) > 0.7:
		# print("Speech FOUND")
		return True
	else:
		# print("Speech NOT FOUND")
		return False
	# for i, segment in enumerate(segments):
	#     print(Pcm2Float(segment))
	    # path = 'chunk-%002d.wav' % (i,)
	    # print(' Writing %s' % (path,))
	    # write_wave(path, segment, sample_rate)


if __name__ == "__main__":
	if len(sys.argv) == 2:
		# path to filename
		print("Speech in sample:", SpeechInAudio(sys.argv[1]))
		Playback.Playback(file=sys.argv[1])
	elif len(sys.argv) == 4:
		# path to filename, start in seconds, end in seconds
		print("Speech in sample:", SpeechInAudio(sys.argv[1], float(sys.argv[2]), float(sys.argv[3])))
		Playback.Playback(file=sys.argv[1], sim_list=[float(sys.argv[2]), float(sys.argv[3])-float(sys.argv[2])])	#start and end in seconds
