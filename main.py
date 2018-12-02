import numpy as np
import torch
import torch.optim as optim
import math
import random
import os
from model import model
import time

random.seed(42)
np.random.seed(0)
torch.manual_seed(2)

BATCH_SIZE = 100
WORDEMBEDDING_DIM = 256
EPOCHS = 10

def read_input():
	data_in = "data.in"
	data_out = "data.out"

	with open(data_in, encoding='utf-8', errors='ignore') as data_in_file:
		data_inputs = [line.split() for line in data_in_file]
	data_in_file.close()

	with open(data_out, encoding='utf-8', errors='ignore') as data_out_file:
		data_outputs = [line.split() for line in data_out_file]
	data_out_file.close()

	all_data = list(zip(data_inputs, data_outputs))

	train_ratio = 0.9
	train_bound = int(train_ratio * len(all_data))
	train_inputs, train_outputs = zip(*all_data)

	train_data = all_data[:train_bound] 
	test_data = all_data[train_bound:] 
	random.shuffle(train_data)
	random.shuffle(test_data)
	train_inputs, train_outputs = zip(*train_data) # unzips the list, i.e. [(a,b), (c,d)] -> [a,c], [b,d]
	test_inputs, test_outputs = zip(*test_data)
	return train_inputs, train_outputs, test_inputs, test_outputs

def build_indices(train_set):
	tokens = [token for line in train_set for token in line]
	forward_dict = {'UNK': 0}
	i = 1
	for token in tokens:
		if token not in forward_dict:
			forward_dict[token] = i 
			i += 1
	return forward_dict

def encode(data, forward_dict):
	return [list(map(lambda t: forward_dict.get(t,0), line)) for line in data]

if __name__ == '__main__':
	train_inputs, train_outputs, test_inputs, test_outputs = read_input()
	forward_dict = build_indices(train_inputs)
	train_inputs = encode(train_inputs, forward_dict)
	test_inputs = encode(test_inputs, forward_dict)
	m = model(vocab_size = len(forward_dict), hidden_dim = WORDEMBEDDING_DIM, out_dim = 2)
	optimizer = optim.SGD(m.parameters(), lr=1.0)
	minibatch_size = BATCH_SIZE
	num_minibatches = len(train_inputs) // minibatch_size

	for epoch in (range(EPOCHS)):
		# Training
		print("Training epoch " + str(epoch))
		# Put the model in training mode
		m.train()
		start_train = time.time()

		for group in range(num_minibatches):
			predictions = None
			truth = None
			loss = 0
			optimizer.zero_grad()
			for i in range(group * minibatch_size, (group + 1) * minibatch_size):
				input_seq = train_inputs[i]
				gold_output = (torch.tensor([0]) if train_outputs[i] == ['0'] else torch.tensor([1]))
				prediction_vec, prediction = m(input_seq)

				if predictions is None:
					predictions = [prediction_vec]
					truth = [gold_output] 
				else:
					predictions.append(prediction_vec)
					truth.append(gold_output)
			loss = m.compute_Loss(torch.stack(predictions), torch.stack(truth).squeeze())
			loss.backward()
			optimizer.step()
		print("Training time: {} for epoch {}".format(time.time() - start_train, epoch))

		# Evaluation
		print("Evaluation")
		# Put the model in evaluation mode
		m.eval()
		start_eval = time.time()

		predictions = 0
		correct = 0
		for input_seq, gold_output in zip(test_inputs, test_outputs):
			_, predicted_output = m(input_seq)
			gold_output = (0 if gold_output == ['0'] else 1)
			correct += int((gold_output == predicted_output))
			predictions += 1
		accuracy = correct / predictions
		print("Evaluation time: {} for epoch {}, Accuracy: {}".format(time.time() - start_eval, epoch, accuracy))

	# Testing
	print("Testing")

	data_in = "test.in"

	with open(data_in, encoding='utf-8', errors='ignore') as data_in_file:
		data_inputs = [line.split() for line in data_in_file]
	data_in_file.close()
	test_inputs = encode(data_inputs, forward_dict)
	start_eval = time.time()
	predictions = []
	# Constructing the output file
	result_file = open("result.csv", "w")
	result_file.write("ID,Label\n")
	i = 0
	for input_seq in test_inputs:
		# Inferencing
		_, predicted_output = m(input_seq)
		predicted_output = (-1 if predicted_output == 0 else predicted_output)
		predictions.append(predicted_output)
		result_file.write(str(i) + "," + str(predicted_output) +"\n")
		i += 1
	print("prediction done")
	print(len(predictions))
	result_file.close()