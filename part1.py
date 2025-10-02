import torch
import torch.nn as nn
import torch.optim as optim
import time
import random
import pickle
import nltk
from nltk.corpus import treebank
import utils 
"""You should not need any other imports."""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nltk.download('treebank')
brown = list(treebank.tagged_sents())

class BiLSTMTagger(nn.Module):
	def __init__(self, data, embedding_dim, hidden_dim):
		super().__init__()
		self.words = utils.Vocab()
		self.tags = utils.Vocab()
		"""TODO: Populate self.words and self.tags as two vocabulary objects using the Vocab class in utils.py.
		This will allow you to easily numberize and denumberize the word vocabulary as well as the tagset.
		Make sure to add <UNK> self.words."""
		self.words.add('<UNK>')
		self.tags.add('<UNK>')

		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim

		for sent in data:
			for w, t in sent:
				self.words.add(w.lower())
				self.tags.add(t)
		"""	TODO: Initialize layers."""
		# embedding
		self.emb = nn.Embedding(len(self.words), self.embedding_dim)
		# lstm
		self.lstm = nn.LSTM(
			input_size=self.embedding_dim,
			hidden_size = self.hidden_dim,
			num_layers=1,
			bidirectional=True,
			batch_first=False,
			dropout=0.0,
		)
		# dropout
		self.dropout = nn.Dropout(p=0.2)
		# W_out
		self.W_out = nn.Linear(2 * self.hidden_dim, len(self.tags))

	def forward(self, sentence):
		"""	TODO: Pass the sentence through the layers of the model. 
			* Because we are using the built-in LSTM, we can pass in an entire sentence rather than iterating through the tokens.
			* IMPORTANT: Because we are dealing with a full sentence now, we have to do minor reshaping. 
				* Before passing the embeddings into the LSTM, we have to do `embeddings.view(len(sentence), 1, -1)`
				* Before passing the LSTM output into dropout, we have to do `lstm_out.view(len(sentence), -1)`
			* Return the output scores from the model (pre-softmax). This will be of shape: len(sentence) x total number of tags, meaning each row corresponds to a word, and the values in each row are the scores for all possible POS tags for that word."""
		embeddings = self.emb(sentence)
		x = embeddings.view(len(sentence), 1, -1)
		lstm_out, _ = self.lstm(x)
		lstm_out = lstm_out.view(len(sentence), -1)
		lstm_out = self.dropout(lstm_out)
		scores = self.W_out(lstm_out)
		return scores 

	def predict(self, scores):
		"""	TODO: Return the most likely tag sequence.
			* When the dim argument is provided, torch.argmax(input, dim) returns a tensor containing the indices of the maximum values along that specified dimension.
			* Since each row of scores corresponds to a different word, and each column corresponds to a different tag, specificy dim=1 (take max along columns)."""
		return torch.argmax(scores, dim=1)

	def fit(self, data, lr=0.01, epochs=5):
		"""	TODO: This function is very similar to fit() from HW1."""
		
		# 1. Initialize the optimizer. Use `torch.optim.Adam` with `self.parameters()` and `lr`.
		optimizer = optim.Adam(self.parameters(), lr=lr)
		
		# 2. Set a loss function variable to `nn.CrossEntropyLoss()`. It includes softmax.
		loss_fn = nn.CrossEntropyLoss()
		
		# 3. Loop through the specified number of epochs.
		for ep in range(1, epochs+1):
		
		#	 1. Put the model into training mode using `self.train()`.
			t0 = time.time()
			self.train()
		
		#	 2. Shuffle the training data using random.shuffle().
			random.shuffle(data)
		
		#	 3. Initialize variables to keep track of the total loss (`total_loss`) and the total number of tokens (`total_tokens`).
			total_loss = 0.0
			total_tokens = 0
		
		#	 4. Loop over each sentence in the training data.
			for sent in data:
		
		#	 	1. Produce a numberized sequence of the words in the sentence. Make words lowercase first, and convert the sequence to a tensor using something like: `torch.tensor(idxs, dtype=torch.long)`.
				words = [w for (w, _) in sent]
				tags = [t for (_, t) in sent]

				idxs = [self.words.numberize(w.lower()) for w in words]
				sentence = torch.tensor(idxs, dtype=torch.long, device=device)


		#		2. Prepare the target labels using something like: `targets = torch.tensor([self.tags.numberize(t) for t in tags], dtype=torch.long)`
				targets = torch.tensor([self.tags.numberize(t) for t in tags],
						   dtype=torch.long, device=device)
		
		#	 	3. Call `self.zero_grad()` to clear any accumulated gradients from the previous update.
				self.zero_grad()
		
		#	 	4. Pass the prepared sequence into the model by doing `self(sentence)` to obtain scores. This automatically calls forward().
				scores = self(sentence)

		#		5. Calculate loss, passing in the output scores and the true target labels.
				loss = loss_fn(scores, targets)
		
		#	 	6. Call `loss.backward()` to compute gradients.
				loss.backward()

		#		7. Apply gradient clipping to prevent exploding gradients. Use `torch.nn.utils.clip_grad_norm_()` with `self.parameters()` and a `max_norm` of 5.0.
				torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)

		#		8. Call `optimizer.step()` to update the model parameters using the computed gradients.
				optimizer.step()

		#		9. Add `loss.item() * len(targets)` to `total_loss`.
				total_loss += loss.item() * len(targets)

		#		10. Add `len(targets)` to `total_tokens`.
				total_tokens += len(targets)

		#	5. Compute the average loss per token by dividing `total_loss / total_tokens`.
			avg_loss = total_loss / max(1, total_tokens)

		#	6. For debugging, it will be helpful to print the average loss per token and the runtime after each epoch. Average loss per token should always decrease epoch to epoch.
			print(f"Epoch {ep:02d} | avg loss/token: {avg_loss:.4f} | {time.time()-t0:.1f}s")

	def evaluate(self, data):
		"""	TODO: Iterating over the sentences in the data, calculate POS tagging accuracy. 
			* Use `self.eval()` and `with torch.no_grad()` so that the model is not trained during evaluation.
			* Prepare the sequence and target labels as in fit().
			* Use self.predict() to get the predicted tags, and then check if it matches the real next character found in the data.
			* Divide the total correct predictions by the total number of tokens to get the final accuracy."""
		self.eval()
		total_correct = 0
		total_tokens = 0
		with torch.no_grad():
			for sent in data:
				words = [w for (w, _) in sent]
				tags = [t for (_, t) in sent]

				idxs = [self.words.numberize(w.lower()) for w in words]
				sentence = torch.tensor(idxs, dtype=torch.long, device=device)
				targets = torch.tensor([self.tags.numberize(t) for t in tags],
						   dtype = torch.long, device=device)
				scores = self(sentence)
				pred = self.predict(scores)

				total_correct += int((pred == targets).sum().item())
				total_tokens += len(tags)
		return total_correct / max(1, total_tokens)

if __name__ == '__main__':
	"""TODO: (reference HW1 part3.py)
	* Use read_pos_file() from utils.py to read train.pos, val.pos, and test.pos.
	* Initialize the model with training data, embedding dim 128, and hidden dim 256.
	* Train the model, calling fit(), on the training data.
	* Test the model, calling evaluate(), on the validation and test data.
	* Predict outputs for the first ten examples in test.pos.
	* Remove all instances of `raise NotImplementedError`!
	"""
	train_sents = utils.read_pos_file('data/train.pos')
	val_sents = utils.read_pos_file('data/dev.pos')
	test_sents = utils.read_pos_file('data/test.pos')

	train_sents += brown 

	model = BiLSTMTagger(train_sents, embedding_dim=128, hidden_dim=256).to(device)
	model.fit(train_sents, lr=0.01, epochs=5)

	print("Dev accuracy:", model.evaluate(val_sents))
	print("Test accuracy:", model.evaluate(test_sents))

	torch.save({
		'model_state_dict': model.state_dict(),
		'words': model.words,
		'tags': model.tags,
		'embedding_dim': 128,
		'hidden_dim': 256,
	}, 'bilstm_pos.pth')

	model.eval()
	with torch.no_grad():
		for i, sent in enumerate(test_sents[:10], 1):
			words = [w for (w, _) in sent]
			idxs = [model.words.numberize(w.lower()) for w in words]
			sentence = torch.tensor(idxs, dtype=torch.long, device=device)

			scores = model(sentence)
			pred_idx = model.predict(scores).tolist()
			pred_tags = [model.tags.denumberize(j) for j in pred_idx]

			print(f"[{i:02d}] Words: {words}")
			print(f"	Pred: {pred_tags}")
