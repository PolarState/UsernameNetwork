#!/usr/bin/env python3

'''TODO:
- add epochs, test, and validation steps.
- add ability to save network.
- refactor to more logical blocks.
'''

import os
import re
import json
import math
import torch
import hashlib
import pickle
import inspect
import numpy

from torch import nn
from torch import optim
from random import shuffle
from torch.utils.data import Dataset, DataLoader

from CharacterLSTM import CharacterLSTM
from CharacterDataset import CharacterDataset

def saveDictionaryAsJSON(dictionary, filename):
	'''
	save dictionary as json to file specified. If the file or directories do
		not exist they will be created.

	Args: 
		dictionary - Python dictionary object, will be saved as json
		filename - full path to save file

	'''

	if not os.path.exists(os.path.dirname(filename)):
		path = os.path.dirname(filename)
		missing_directories = []
		while path:
			path, dirname = os.path.split(path)
			if not os.path.exists(path):
				missing_directories.insert(0, dirname)

		print(missing_directories)
		path = '.'
		for d in missing_directories:
			path = path + '/' + d
			os.mkdir(path)

	f = open(filename, "w+")
	f.write(json.dumps(dictionary))


def createCharacterCountDictionary(text):
	'''
	create dictionary of characters from string.

	Args:
		text - to create dictionary from

	Returns:
		dictionary with key of characters and value of # of occurances in text
	'''

	# create a dictionary of characters used
	char_dict = {}
	for c in text:
		if not c in char_dict:
			char_dict[c] = 1
		else:
			char_dict[c] = char_dict[c] + 1

	return(char_dict)






def inputTensor(line, vocab_list):
	'''
	input tensors exclude the last character of line and are encoded
		with the index of the symbol in the vocabList
	'''

	# chop off last letter
	input_line = line[0:-1]

	# encode with index of symbol in dictionary
	input_symbols = [vocab_list.index(s) for s in input_tensor]

	return torch.LongTensor(input_symbols)


def outputTensor(line, vocab_list):
	'''
	output tensors exclude the first symbol of line and are encoded
		with the index of the symbol in the vocabList
	'''

	# chop off first letter
	output_line = line[1:]

	# encode with index of symbol in dictionary
	output_symbols = [vocab_list.index(s) for s in output_tensor]

	return torch.LongTensor(output_symbols)


if __name__ == '__main__':

	vocab = {}
	if os.path.exists('cache/characterDict.json'):
		with open('cache/characterDict.json') as f:
			vocab = json.loads(f.read())

	else:
		# read file with usernames and create a list
		with open('../UsernameData/usernames.txt', "r") as f:
			text = f.read()

		# pass list to dictionary creator
		vocab = createCharacterCountDictionary(text)
		saveDictionaryAsJSON(vocab, 'cache/characterDict.json')


	# open file and load text
	with open('../UsernameData/usernames.txt', "r") as f:
		corpus = f.read()


	corpus_hash = hashlib.sha224(corpus.encode('utf-8')).hexdigest()
	if os.path.exists(f'cache/character-dataset-{corpus_hash}.pickle'):
		print('found cache')
	else:
		dataset = CharacterDataset(corpus, r'(.+\n)')
	
	# hash class based on it's member functions:
	# print(f'CharacterDataset: {inspect.getmembers(CharacterDataset, predicate=inspect.ismethod)}')
	# print(f'{dataset.methods}')
	# https://stackoverflow.com/questions/1911281/how-do-i-get-list-of-methods-in-a-python-class


	# dasetfile = open('cache/character-dataset-{corpus_hash}.pickle', 'wb')
	# pickle.dump(dataset, dasetfile)
	# dasetfile.close()

	batch_size = 10

	characterDataloader = DataLoader(
			dataset = dataset,
			batch_size = batch_size)

	print(f'hash a function: {hash(dataset.vocabLength)}')

	# define model
	model = CharacterLSTM(input_dim=dataset.vocabLength(),
					output_dim=dataset.vocabLength(),
					hidden_dim=512,
					batch_size=batch_size)

	# define optimizer and scheduler
	optimizer = optim.Adam(model.parameters(), lr=0.001)
	loss_function = nn.CrossEntropyLoss(ignore_index=0)

	for inputs, labels, input_length in characterDataloader:

		model.zero_grad()

		# print(input_length)
		outputs = model(inputs, input_length)

		# reshape output to shape: [elements, element_size] (one-hot-encoding length)
		outputs_flat = outputs.view(-1, dataset.vocabLength())
		# reshape labels into: [elements] (elements for labels are not one-hot-encoded)
		labels_flat = labels.view(-1)

		loss = loss_function(outputs_flat, labels_flat)
		print(loss)
		loss.backward()

		optimizer.step()

		

