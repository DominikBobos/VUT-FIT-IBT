import os, sys, glob                    # for file path, finding files, etc
import shutil

parent_dir = "/home/dominik/Desktop/bak/dev_data"

files = glob.glob(parent_dir + '/eval_goal/*.wav')
for idx, file in enumerate(files):
	filename = file.split('/')[-1][:-4]
	print(idx+1, '/', len(files), 'Processing: ', filename) 
	if os.path.exists(parent_dir + '/train_goal/train_goal_phn/' + filename + '.lin'):
		shutil.move(parent_dir + '/train_goal/train_goal_phn/' + filename + '.lin', parent_dir + '/eval_goal/eval_goal_phn/' + filename + '.lin')
		shutil.move(parent_dir + '/train_goal/train_goal_str/' + filename + '.txt', parent_dir + '/eval_goal/eval_goal_str/' + filename + '.txt')
		shutil.move(parent_dir + '/train_goal/train_goal_lat/' + filename + '.latt', parent_dir + '/eval_goal/eval_goal_lat/' + filename + '.latt')
		print(filename,"'s features was moved succesfully.")
		
files = glob.glob(parent_dir + '/eval_clear/*.wav')
for idx, file in enumerate(files):
	filename = file.split('/')[-1][:-4]
	print(idx+1, '/', len(files), 'Processing: ', filename) 
	if os.path.exists(parent_dir + '/train_clear/train_clear_phn/' + filename + '.lin'):
		shutil.move(parent_dir + '/train_clear/train_clear_phn/' + filename + '.lin', parent_dir + '/eval_clear/eval_clear_phn/' + filename + '.lin')
		shutil.move(parent_dir + '/train_clear/train_clear_str/' + filename + '.txt', parent_dir + '/eval_clear/eval_clear_str/' + filename + '.txt')
		shutil.move(parent_dir + '/train_clear/train_clear_lat/' + filename + '.latt', parent_dir + '/eval_clear/eval_clear_lat/' + filename + '.latt')
		print(filename,"'s features was moved succesfully.")