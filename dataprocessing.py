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

def createUniqueCharacterList(text):
	'''
	create a list of unique characters that appear in text

	Args:
		text - to sample for unique characters from

	Returns:
		list of unique characters from text
	'''
	print('__createUniqueCharacterList')

	# create a dictionary of characters used
	unique_char_list = ['<pad>'] + sorted(set(text))# ensure that element 0 is padding

	print(f'len(unique_char_list): {len(unique_char_list)}')
	return(unique_char_list)


def createPaddedSequenceList(text, delimiter, character_encoding):
	'''
	create list of sequences from the original text where a sequence is a
		logical group of characters such as a paragraph, sentence, or name
		in a list of names.

	Args:
		text - original text to create sequences from
		delimiter - character or characters which delimit the sequences
		character_encoding - map of character indexes for one-hot-encoding

	Returns:
		list of strings where each string is a 'sequence'
	'''
	print('__createPaddedSequenceList')
	print('filter')
	sequence_list = list(filter(lambda x: x != '', re.split(delimiter, text)))
	print('length tensor')
	sequence_length_list = torch.LongTensor([len(s) for s in sequence_list])
	print('empty tensor')
	padded_sequence_list = torch.zeros((len(sequence_list), sequence_length_list.max())
										, dtype=torch.long)
	print('fill tensor')
	for idx, (seq, seq_len) in enumerate(zip(sequence_list, sequence_length_list)):
		vectorized_list = [character_encoding.index(c) for c in seq]
		padded_sequence_list[idx, :seq_len] = torch.LongTensor(vectorized_list)

	return (padded_sequence_list, sequence_length_list)


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


	# corpus_hash = hashlib.sha224(corpus.encode('utf-8')).hexdigest()
	# if os.path.exists(f'cache/character-dataset-{corpus_hash}.pickle'):
	# 	print('found cache')
	# else:
	
	# Get list of unique characters, used for one-hot-encoding.
	unique_char_list = createUniqueCharacterList(corpus)
	# Transform corpus of usernames into a list of padded usernames with their original lengths preserved
	padded_seq_list, padded_seq_length_list = createPaddedSequenceList(corpus, r'(.+\n)', unique_char_list)

	# Split into seperate validation and training sets.
	padded_seq_tuple_list = list(zip(padded_seq_list, padded_seq_length_list))
	shuffle(padded_seq_tuple_list)

	padded_seq_list, padded_seq_length_list = [list(d) for d in zip(*padded_seq_tuple_list)]

	validation_set_size = int(len(padded_seq_list)/10)
	
	val_padded_seq_list = padded_seq_list[:validation_set_size]
	val_padded_seq_length_list = padded_seq_length_list[:validation_set_size]
	train_padded_seq_list = padded_seq_list[validation_set_size:]
	train_padded_seq_length_list = padded_seq_length_list[validation_set_size:]
	
	dataset_val = CharacterDataset(unique_char_list, val_padded_seq_list, val_padded_seq_length_list)
	dataset_train = CharacterDataset(unique_char_list, train_padded_seq_list, train_padded_seq_length_list)
	
	# hash class based on it's member functions:
	# print(f'CharacterDataset: {inspect.getmembers(CharacterDataset, predicate=inspect.ismethod)}')
	# print(f'{dataset.methods}')
	# https://stackoverflow.com/questions/1911281/how-do-i-get-list-of-methods-in-a-python-class


	# dasetfile = open('cache/character-dataset-{corpus_hash}.pickle', 'wb')
	# pickle.dump(dataset, dasetfile)
	# dasetfile.close()

	batch_size = 2048

	trainingCharDataloader = DataLoader(
			dataset = dataset_train,
			batch_size = batch_size,
			shuffle = True
		)
	
	valCharDataloader = DataLoader(
			dataset = dataset_val,
			batch_size = batch_size,
			shuffle = True
		)

	# print(f'hash a function: {hash(dataset.vocabLength)}')

	# define model
	model = CharacterLSTM(input_dim=len(unique_char_list),
					output_dim=len(unique_char_list),
					hidden_dim=512,
					batch_size=batch_size)

	# define optimizer and scheduler
	optimizer = optim.Adam(model.parameters(), lr=0.001)
	loss_function = nn.CrossEntropyLoss(ignore_index=0)

	print("Training Start")
	for e in range(20):
		val_loss = 0
		train_loss = 0

		for i, (inputs, labels, input_length) in enumerate(trainingCharDataloader):

			optimizer.zero_grad()

			# print(input_length)
			outputs = model(inputs, input_length)

			# reshape output to shape: [elements, element_size] (one-hot-encoding length)
			outputs_flat = outputs.view(-1, len(unique_char_list))
			# reshape labels into: [elements] (elements for labels are not one-hot-encoded)
			labels_flat = labels.view(-1)

			train_loss = loss_function(outputs_flat, labels_flat)
			train_loss.backward()

			optimizer.step()

			print(f"{i*batch_size/len(dataset_train)*100}% idx: {idx}", end='\r')

		for inputs, labels, input_length in valCharDataloader:

			outputs = model(inputs, input_length)

			# reshape output to shape: [elements, element_size] (one-hot-encoding length)
			outputs_flat = outputs.view(-1, len(unique_char_list))
			# reshape labels into: [elements] (elements for labels are not one-hot-encoded)
			labels_flat = labels.view(-1)

			val_loss = loss_function(outputs_flat, labels_flat)
		
		print(f"Train loss: {train_loss}")
		print(f"Val loss: {val_loss}")

		torch.save(model.state_dict, f"./test_model_{i:02}")


		

