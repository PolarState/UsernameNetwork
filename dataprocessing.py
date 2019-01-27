#!/usr/bin/env python3

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

class CharacterDataset(Dataset):

	def __init__(self, corpus, delimiter):

		# list of unique characters for one-hot-encoding
		self.character_encoding = self.__createUniqueCharacterList(corpus)

		# list of sequences, where a sequence is whatever logical group 
		#	of characters and could be a sentence, word, or paragraph.
		self.sequence_list = self.__createPaddedSequenceList(corpus, delimiter)
		
		# length in characters of entire dataset corpus
		# self.lenth = sum([len(s) for s in self.sequence_list])

		# prepare for first sampling
		# self.__reshuffle()


	def vocabLength(self):
		return len(self.character_encoding)

	def sequenceLength(self):
		return len(self.sequence_list[0])

	def __reshuffle(self):
		# if we need to 
		print('__reshuffle')

		shuffle(self.sequence_list)

		self.shuffled_text = ''.join(self.sequence_list)

	def __createUniqueCharacterList(self, text):
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


	def __createPaddedSequenceList(self, text, delimiter):
		'''
		create list of sequences from the original text where a sequence is a
			logical group of characters such as a paragraph, sentence, or name
			in a list of names.

		Args:
			text - original text to create sequences from
			delimiter - character or characters which delimit the sequences

		Returns:
			list of strings where each string is a 'sequence'
		'''
		print('__createPaddedSequenceList')
		print('filter')
		sequence_list = list(filter(lambda x: x != '', re.split(delimiter, text)))
		print('length tensor')
		self.sequence_length_list = torch.LongTensor([len(s) for s in sequence_list])
		print('empty tensor')
		padded_sequence_list = torch.zeros((len(sequence_list), self.sequence_length_list.max())
											, dtype=torch.long)
		print('fill tensor')
		for idx, (seq, seq_len) in enumerate(zip(sequence_list, self.sequence_length_list)):
			vectorized_list = [self.character_encoding.index(c) for c in seq]
			padded_sequence_list[idx, :seq_len] = torch.LongTensor(vectorized_list)

		return padded_sequence_list


	def __sequenceToOneHotEncodng(self, sequence):

		encodedTensor = torch.zeros(len(sequence), len(self.character_encoding))
		for idx, c in enumerate(sequence):
			encodedTensor[idx][c] = 1

		return encodedTensor


	def __characterToOneHotEncoding(self, character):

		encodedTensor = torch.zeros(len(self.character_encoding))
		encodedTensor[self.character_encoding.index(character)] = 1
		
		return encodedTensor


	def __characterToIdx(self, character):
		return torch.tensor(self.character_encoding.index(character), dtype=torch.long)


	def __len__(self):
		return len(self.sequence_list) - 1


	def __getitem__(self, idx):
		if idx < len(self) - 1:

			# translate into encoding

			# copy shifted sequence from sequence_list
			input_seq = self.sequence_list[idx][0:-1].clone()
			label_seq = self.sequence_list[idx][1:].clone()

			# sequence list is already padded so remove last character from input sequence
			input_seq[self.sequence_length_list[idx] - 1] = 0

			# apply one hot encoding to the sequences (but not labels)
			inputs = torch.tensor(self.__sequenceToOneHotEncodng(input_seq), dtype=torch.float)

			input_length = self.sequence_length_list[idx]

			return inputs, label_seq, input_length

class CharLSTM(nn.Module):

	def __init__(self, input_dim, output_dim, hidden_dim, batch_size):
		
		super(CharLSTM, self).__init__()
		self.hidden_dim = hidden_dim
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.batch_size = batch_size

		self.init_hidden()

		self.lstm = nn.LSTM(input_size=input_dim, hidden_size=self.hidden_dim)

		self.linear = nn.Linear(hidden_dim, output_dim)

		self.softmax = nn.Softmax()

	def forward(self, inputs, input_lengths):
		# lstm_inputs = inputs.view(len(inputs), 1, len(inputs[0]))

		# Reset if the next sequence should not be thought of as a continuation of a sequence.
		self.init_hidden()

		# print(f'inputs.size() {inputs.size()}')

		# extract dimentions of inputs
		batch_size, seq_len, _ = inputs.size()

		# sort the inputs, (this is a requirement of the pack_padded_sequence and pad_packed_sequence functions)
		# and store the indexes to restore the original order
		lengths_sorted, perm_idx = input_lengths.sort(0, descending=True)
		inputs_sorted = inputs[perm_idx]

		# pack inputs to feed into LSTM
		inputs_packed = torch.nn.utils.rnn.pack_padded_sequence(inputs_sorted, lengths_sorted, batch_first=True)

		# forward pass in LSTM
		lstm_outputs, self.hidden = self.lstm(inputs_packed, self.hidden)

		# pad sequence
		lstm_outputs_padded, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_outputs, total_length=seq_len, batch_first=True)

		# print(f'lstm_outputs_padded.size() {lstm_outputs_padded.size()}')

		# transform dimentions of output for linear layer
		lstm_outputs_padded = lstm_outputs_padded.contiguous()
		lstm_outputs_padded = lstm_outputs_padded.view(-1, lstm_outputs_padded.size()[2])

		# print(f'lstm_outputs_padded.size() {lstm_outputs_padded.size()}')

		linear_outputs = self.linear(lstm_outputs_padded)

		# print(f'linear_outputs.size() {linear_outputs.size()}')

		softmax_outputs = self.softmax(linear_outputs)

		# print(f'softmax_outputs.size() {softmax_outputs.size()}')

		outputs = softmax_outputs.view(batch_size, seq_len, self.output_dim)
		
		# unsort batch
		_, unperm_idx = perm_idx.sort(0)
		outputs = outputs[unperm_idx]

		return outputs

	def init_hidden(self):
		# hidden is an embedding. an array has to be created to store the embedding.

		# print(self.hidden_dim)
		self.hidden = (torch.zeros(1, self.batch_size, self.hidden_dim),
					torch.zeros(1, self.batch_size, self.hidden_dim))


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
	model = CharLSTM(input_dim=dataset.vocabLength(),
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

		# print(outputs)

