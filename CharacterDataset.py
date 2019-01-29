import re
import torch

from torch.utils.data import Dataset

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
