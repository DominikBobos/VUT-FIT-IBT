from subprocess import call
import argparse                         # for parsing arguments
import os, sys, glob                    # for file path, finding files, etc

"""''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Arguments parsing
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''"""
parser = argparse.ArgumentParser()
parser.add_argument("--src", help="source directory path")                            # directory to import
parser.add_argument("--dst", help="destination directory path")                       # directory to export
parser.add_argument("--string", action="store_true", help="Export phonemes to txt") 
arguments = parser.parse_args()


if not arguments.src:
	src = os.getcwd()
else:
	if os.path.isdir(arguments.src):
		src = arguments.src
	else:
		sys.stderr.write("Invalid source directory.\n")
		sys.exit(1)
if not arguments.dst:
	dst = os.getcwd()
else:
	if os.path.isdir(arguments.dst):
		dst = arguments.dst
	else:
		try:
			os.mkdir(arguments.dst)                                  # create destination directory
			dst = arguments.dst
		except:
			sys.stderr.write("Invalid destination directory.\n")
			sys.exit(1)


def get_posteriors():
	files = glob.glob(src+'*.wav')
	if os.path.isfile("set_phnrec") == False:
		sys.stderr.write("set_phnrec script is not found")
		sys.exit(1)

	for idx, file in enumerate(files):
		src_file = file
		print("{}/{} (file: {})".format(idx,len(files), file.split('/')[-1]))
		if arguments.string:
			dst_file = dst + file.split('/')[-1].replace('.wav', '.txt')
			if os.path.exists(dst_file):
				continue
			call(""". /home/dominik/Desktop/bak/scripts/VUT-FIT-IBT/scripts/set_phnrec
phnrec -c $phnrecdir/PHN_CZ_SPDAT_LCRC_N1500/ -w lin16 -t {} -i "{}" -o "{}"
			""".format("str", src_file, dst_file), shell=True)
		else:
			dst_file = dst + file.split('/')[-1].replace('.wav', '.lin')
			if os.path.exists(dst_file):
				continue
			call(""". /home/dominik/Desktop/bak/scripts/VUT-FIT-IBT/scripts/set_phnrec
phnrec -c $phnrecdir/PHN_CZ_SPDAT_LCRC_N1500/ -w lin16 -t {} -i "{}" -o "{}"
			""".format("post", src_file, dst_file), shell=True) 

get_posteriors()