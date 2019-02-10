import torch

from torch import nn

class CharacterLSTM(nn.Module):

	def __init__(self, input_dim, output_dim, hidden_dim, batch_size):
		
		super(CharacterLSTM, self).__init__()
		self.hidden_dim = hidden_dim
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.batch_size = batch_size

		self.init_hidden()

		self.lstm = nn.LSTM(input_size=input_dim, hidden_size=self.hidden_dim)

		self.linear = nn.Linear(hidden_dim, output_dim)

		self.softmax = nn.Softmax(dim=1)

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
		inputs_packed = nn.utils.rnn.pack_padded_sequence(inputs_sorted, lengths_sorted, batch_first=True)

		# forward pass in LSTM
		lstm_outputs, self.hidden = self.lstm(inputs_packed, self.hidden)

		# pad sequence
		lstm_outputs_padded, _ = nn.utils.rnn.pad_packed_sequence(lstm_outputs, total_length=seq_len, batch_first=True)

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
