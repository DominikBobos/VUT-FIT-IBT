import pyaudio
import wave
import time

#currently just for two files playback
def Playback(files, sim_list):
	# set desired values

	for similarity in sim_list:
		for idx, file in enumerate(files):
			start = similarity[0, idx] / 100
			length = (similarity[-1, idx] - similarity[0, idx])/100
			# open wave file
			wave_file = wave.open(file, 'rb')

			# initialize audio
			py_audio = pyaudio.PyAudio()
			stream = py_audio.open(format=py_audio.get_format_from_width(wave_file.getsampwidth()),
								   channels=wave_file.getnchannels(),
								   rate=wave_file.getframerate(),
								   output=True)

			# skip unwanted frames
			n_frames = int(start * wave_file.getframerate())
			wave_file.setpos(n_frames)

			# write desired frames to audio buffer
			n_frames = int(length * wave_file.getframerate())
			frames = wave_file.readframes(n_frames)
			stream.write(frames)

			# close and terminate everything properly
			stream.close()
			py_audio.terminate()
			wave_file.close()

			sys.stdout.write('\r\a{i}'.format(i=1))
			sys.stdout.flush()
			time.sleep(1)
			sys.stdout.write('\n')