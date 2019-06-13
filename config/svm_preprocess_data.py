
import os
import xml.etree.ElementTree as etree

le_closed_idx = 3
re_closed_idx = 5
num_tokens_per_row = 19
data = []
labels = []
dirname = os.getcwd()

def is_comment(line):
	return line.startswith("#") or line.startswith("\n")

def process_ears(ears_path):
	ears = []
	tree = etree.parse(ears_path)
	root = tree.getroot()
	vector = root[0]
	ear_cnt = int(vector[0].text)
	for i in range(len(vector) - ear_cnt, len(vector)):
		ears.append(float(vector[i].text))
	return ears

def process(folder):
	blinks = []
	ears_path = os.path.join(dirname, folder + "/" + folder + ".xml")
	tag_path = os.path.join(dirname, folder +"/" + folder + ".tag")
	#EARs currently produced by OpenArk BlinkDetector, files manually created by running C++ script.	
	ears = process_ears(ears_path)
	#Parse for blinks from annotations of Eyeblink8 Dataset.
	with open(tag_path, "r") as f:
		line_cnt = 0
		for line in f:
			if is_comment(line):
				print("Skipping comment.")
				continue
			tokens = line.split(':')
			assert (len(tokens) == num_tokens_per_row), "Line {} invalid. Contains {} tokens only.".format(str(line_cnt), str(len(tokens)));
			if tokens[le_closed_idx] == 'C' or tokens[re_closed_idx] == 'C':
				blinks.append(1)
			else: 
				blinks.append(0)
			line_cnt += 1
	# Cech Paper Suggestion: 13-dimensional feature vectors. +/- 6 frames from current (assuming 30FPS video).
	# TODO: Check performance of skipping initial 6 / last 6 frames vs. padding with mean values or 0s.
	assert (len(ears) == len(blinks)), "# of EARs: {} and # of blinks: {}. Lengths do not match.".format(len(ears), len(blinks))
	for i in range(len(ears)):
		if i < 6 or i > (len(ears) - 7):
			continue;
		feat_vec = []
		for j in range(i-6, i+7):
			feat_vec.append(ears[j])
		data.append(feat_vec)
		print(feat_vec)
		labels.append(blinks[i])
		print(blinks[i])		

process("1")
print(data)
print(labels)
#for i in range(8):




"""
Pseudocode: 
1) Read in a 

"""