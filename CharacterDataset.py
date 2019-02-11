# TODO:
# 
# Questions:
# What is best practice for creating datasets for training/valudation/test?
# 


import re
import torch

from torch.utils.data import Dataset

class CharacterDataset(Dataset):

	def __init__(self, character_encoding, sequence_list, sequence_length_list):
		'''
		Args: 
			character_encoding - 
			sequence_list - 
		'''

		# list of unique characters for one-hot-encoding
		self.character_encoding = character_encoding

		# list of sequences, where a sequence is whatever logical group 
		#	of characters and could be a sentence, word, or paragraph.
		self.sequence_list = sequence_list

		# list of original word lengths.
		self.sequence_length_list = sequence_length_list

	def vocabLength(self):
		return len(self.character_encoding)

	def sequenceLength(self):
		return len(self.sequence_list[0])

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
		if idx < len(self):

			# translate into encoding

			# copy shifted sequence from sequence_list
			input_seq = self.sequence_list[idx][0:-1].clone()
			label_seq = self.sequence_list[idx][1:].clone()

			# sequence list is already padded so remove last character from input sequence
			# -2 because the sequence has already been shotened by the slice operation above.
			input_seq[self.sequence_length_list[idx] - 2] = 0

			# apply one hot encoding to the sequences (but not labels)
			# inputs = torch.tensor(self.__sequenceToOneHotEncodng(input_seq), dtype=torch.float)
			inputs = self.__sequenceToOneHotEncodng(input_seq).clone().detach()

			input_length = self.sequence_length_list[idx]

			return inputs, label_seq, input_length

