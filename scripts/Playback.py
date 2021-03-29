import pyaudio
import wave
import time
import sys
import pickle
from beepy import beep


def GetOneFolderUp(file):
	processed = '' 
	folders = file.split('/')
	for idx, folder in enumerate(folders):
		if idx == len(folders)-2: #the last but one item (folder to get away from)
			continue
		if idx == len(folders)-1: #the last one (file)
			processed += folder #filename
			continue
		processed += folder + '/'
	return processed

def GetAudioFile(file):
	if file[-4:] == '.wav':
		return file
	if file[-4:] == '.lin' or file[-4:] == '.fea':
		processed = GetOneFolderUp(file)[:-4] + '.wav'
		return processed


def Play(file, start=0.0, length=5.0):
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


def FramesToSeconds(frames, frame_reduction):
	start = frames[0][0] * frame_reduction
	end = frames[-1][0] * frame_reduction
	return start, end, end - start 


def Playback(files=None, sim_list=None, file_list=None, file=None):
	if file is not None:
		file = GetAudioFile(file)
		print("Playing:{}".format(file.split('/')[-1]))
		if sim_list is not None:
			Play(file, sim_list[0], sim_list[1])
		else:
			Play(file)

	if file_list is not None:
		FA_only = True
		frames_threshold = 450
		try:
			open_file = open(file_list, "rb")
			pkl_list = pickle.load(open_file)
			open_file.close()
			print("{} found. Playback will start shortly".format(file_list))
		except IOError:
			print("No {} exist. No playback will be performed".format(file_list))
		for element in pkl_list:
			if element[1] == []:
				print("Skipping {}, no frames to play".format(element[0].split('/')[-1]))
				continue
			if FA_only and len(element[0].split('/')[-1]) > 30: 
				print("Skipping {}, Mode 'FA only's is on. Hits are not played".format(element[0].split('/')[-1]))
				continue
			if FA_only and len(element[1])*element[2] < frames_threshold:
				print("Skipping {}, Mode 'FA only' is on. Short FA are not played. (shorter than {})".format(element[0].split('/')[-1], frames_threshold))
				continue
			file = GetAudioFile(element[0])
			frame_reduction = element[2]
			start, end, length = FramesToSeconds(element[1], frame_reduction)
			print("Playing:{} Frames:{}-{}".format(file.split('/')[-1], start, end))
			length /= 100
			start /= 100
			Play(file, start, length) 
			beep(sound='coin')

	if files is not None:
		for idx_sim, similarity in enumerate(sim_list):
			for idx, file in enumerate(files):
				file = GetAudioFile(file)
				print("Playing:{} Frames:{}-{} Part:{}/{}".format(file.split('/')[-1], similarity[0, idx], similarity[-1, idx], idx_sim+1, len(sim_list)))
				start = similarity[0, idx] / 100
				length = (similarity[-1, idx] - similarity[0, idx])/100
				Play(file, start, length)
				#1 : 'coin', 2 : 'robot_error', 3 : 'error', 4 : 'ping', 5 : 'ready', 6 : 'success', 7 : 'wilhelm'
				beep(sound='error')
			print("Next similarity")
			beep(sound='coin')

if __name__ == "__main__":
	if sys.argv[1][-3:] == "pkl":
		Playback(file_list=sys.argv[1])
	else:
		if len(sys.argv) == 4:
			Playback(file=sys.argv[1], sim_list=[float(sys.argv[2]), float(sys.argv[3])])
		else:
			Playback(file=sys.argv[1])