# HMM_sequence_detection

The HMM is a statistical model that assumes the system being modeled is a Markov process with unobserved (hidden) states.

## Formulation

A Hidden Markov Model (HMM) is a statistical Markov model in which the system being modeled is assumed to be a Markov process with unobserved (hidden) states. An HMM can be characterized by the following:

- N, the number of states in the model. The states are {S1, S2, ..., SN} and the state at time t is denoted as qt.

- M, the number of distinct observation symbols per state, i.e., the discrete alphabet size. The observation symbols correspond to the physical output of the system being modeled. They are {v1, v2, ..., vM}.

- The state transition probability distribution A, where a[i][j] is the probability of moving from state i to state j. This can be represented as:

```
A = {a[i][j]} = P(qt = Sj | qt-1 = Si), 1 <= i, j <= N
```

- The observation symbol probability distribution B, where b[j][k] is the probability of observing symbol vk in state Sj. This can be represented as:

```
B = {b[j][k]} = P(Ot = vk | qt = Sj), 1 <= j <= N, 1 <= k <= M
```

- The initial state distribution π, where π[i] is the probability of starting in state Si. This can be represented as:

```
π = {π[i]} = P(q1 = Si), 1 <= i <= N
```

## Step-by-step Explanation of How the Algorithm Works

- **Importing Libraries**: We import the necessary libraries, including NumPy for numerical computations and the `hmmlearn` library for Hidden Markov Models.

- **Define sequences**: The script defines several sequences of observations. Each sequence is represented as a list of word indices. These sequences are used to train the HMM.

- **Prepare data for training**: The script combines all sequences into a single list and calculates the lengths of the sequences. It then reshapes the sequences into a 2D array, which is the input format required by hmmlearn.

- **Define and train the HMM**: The script defines a GaussianHMM with a specified number of hidden states and trains it on the prepared data.

- **Evaluate new sequences**: The script defines a function to evaluate the log-likelihood of a new sequence given the trained HMM. It then uses this function to evaluate several new sequences.

### Recommended library versions

numpy==1.24.3

hmmlearn==0.3.2
