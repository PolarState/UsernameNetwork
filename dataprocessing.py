#!/usr/bin/env python3

import os
import json
import math
import torch

from random import shuffle
from torch.utils.data import Dataset

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

	def __init__(self, text, delimiter):

		# list of unique characters for one-hot-encoding
		self.character_encoding = self.__createUniqueCharacterList(text)

		# list of sequences, where a sequence is whatever logical group 
		#	of characters and could be a sentence, word, or paragraph.
		self.sequence_list = self.__createSequenceList(text, delimiter)
		
		# length in characters
		self.lenth = sum([len(s) for s in self.sequence_list])

		# prepare for first sampling
		self.__reshuffle()

	def __reshuffle(self):
		print('__reshuffle')

		shuffle(self.sequence_list)

		self.shuffled_text = ''.join(self.sequence_list)
		# for s in self.sequence_list:
		# 	self.shuffled_text += s

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
		unique_char_list = []
		for c in text:
			if not c in unique_char_list:
				unique_char_list.append(c)

		return(unique_char_list)


	def __createSequenceList(self, text, delimiter):
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
		print('__createSequenceList')
		sequence_list = []
		for s in text.split(delimiter):
			sequence_list.append(s + delimiter)

		return sequence_list


	def __lineToOneHotEncodng(self, line):

		encodedTensor = torch.zeros(len(line), len(self.character_encoding))
		for i in range(len(line)):
			encodedTensor[i][self.character_encoding.index(line[i])] = 1

		return encodedTensor


	def __characterToOneHotEncoding(self, character):

		encodedTensor = torch.zeros(len(self.character_encoding))
		encodedTensor[self.character_encoding.index(character)] = 1
		
		return encodedTensor


	def __len__(self):
		return self.lenth


	def __getitem__(self, idx):
		if idx < len(self):

			# translate into encoding
			return self.__characterToOneHotEncoding(self.shuffled_text[idx])

		else:
			self.__reshuffle()
			raise StopIteration()


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


	# create list of words
	with open('../UsernameData/usernames.txt', "r") as f:
		text = f.read()



	# character_encoding = list(vocab.keys())

	# print(character_encoding.index('a'))

	dataset = CharacterDataset(text, '\n')

	for d in dataset:
		print(list(d))
