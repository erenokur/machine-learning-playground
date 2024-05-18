import numpy as np
from hmmlearn import hmm

# Example samples, represented as sequences of word indices
sequences = {
    'sample1': [[0, 1, 2, 3], [1, 2, 3, 4], [0, 2, 3, 4]],
    'sample2': [[0, 1, 2, 3], [1, 2, 3, 4], [0, 2, 3, 4]],
    'sample3': [[0, 1, 2, 3], [1, 2, 3, 4], [0, 2, 3, 4]],
}

# Combine sequences from all sample for training
all_sequences = []
for sample, seq in sequences.items():
    # print("Sample:", sample)
    # print("Sample has nan:", np.isnan(seq).any())
    all_sequences.extend(seq)

# Concatenate sequences and lengths for training
lengths = [len(seq) for seq in all_sequences]
X = np.concatenate([np.array(seq).reshape(-1, 1) for seq in all_sequences])

# Define and train the HMM
n_components = 4
model = hmm.GaussianHMM(n_components=n_components, n_iter=100, random_state=42)
model.fit(X, lengths)

# Function to evaluate if a new sequence matches the trained HMM
def evaluate_sequence(sequence):
    sequence_arr = np.array(sequence).reshape(-1, 1)
    logprob = model.score(sequence_arr)
    return logprob

# Example new sequence to evaluate
new_sequence_1 = [0, 1, 2, 3]
new_sequence_2 = [1, 2, 3, 4]
new_sequence_3 = [2, 3, 4, 5] 

# Likelihood of new sequences in the trained model if result is positive then the sequence is a match
print("Log likelihood of new_sequence_1 in model:", evaluate_sequence(new_sequence_1))
print("Log likelihood of new_sequence_2 in model:", evaluate_sequence(new_sequence_2))
print("Log likelihood of new_sequence_3 in model:", evaluate_sequence(new_sequence_3))
