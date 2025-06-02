# Dynamic Coattention Networks & Question Answering Glossary

## **Core DCN Concepts**

### **Coattention Mechanism**
Attention mechanism that simultaneously attends to both question and document, creating co-dependent representations.
Computes how each word in the question relates to each word in the document and vice versa.

### **Dynamic Coattention Network (DCN)**
End-to-end neural network for question answering that uses iterative refinement to escape local maxima.
Consists of encoders, coattention mechanism, and dynamic decoder.

### **Dynamic Decoder**
Iterative pointing mechanism that alternates between predicting start and end positions of answer spans.
Can revise predictions based on previous estimates, unlike single-pass decoders.

### **Local Maxima**
Suboptimal solutions that appear good locally but aren't globally optimal.
In QA, this means plausible but incorrect answers that traditional models get "stuck" on.

### **Sentinel Vector**
Learnable parameter vector that allows the model to "not attend" to any particular input word.
Acts as a fallback option in attention mechanisms.

## **Model Architecture Components**

### **Affinity Matrix (L)**
Matrix containing similarity scores between all pairs of document and question words.
Computed as L = D^T Q, where D is document encoding and Q is question encoding.

### **Answer Span**
Contiguous sequence of words in the document that constitutes the answer to a question.
Defined by start and end positions.

### **Bidirectional LSTM (BiLSTM)**
LSTM that processes sequences in both forward and backward directions, providing richer contextual representations.

### **Highway Maxout Network (HMN)**
Deep neural network combining Highway Networks and Maxout Networks.
Used in the dynamic decoder to compute start/end position scores for each document word.

### **Pointing Mechanism**
Method for selecting specific positions in the input sequence as answers, rather than generating new words from a vocabulary.

## **Question Answering Fundamentals**

### **Exact Match (EM)**
Evaluation metric that measures the percentage of predictions that match the ground truth answer exactly (character-by-character).

### **Extractive QA**
Question answering where the answer is a substring (span) extracted directly from the given document, as opposed to generating new text.

### **F1 Score**
Evaluation metric measuring the overlap between predicted and ground truth answers, calculated as harmonic mean of precision and recall.

### **Machine Reading Comprehension (MRC)**
AI task where machines must understand text passages and answer questions about them, demonstrating comprehension abilities.

### **SQuAD (Stanford Question Answering Dataset)**
Large-scale dataset containing 100,000+ questions posed by humans on Wikipedia articles, where answers are text spans.

## **Attention Mechanisms**

### **Attention Visualization**
Technique for understanding model behavior by displaying attention weights as heatmaps showing which input parts the model focuses on.

### **Attention Weights**
Probability distributions over input elements indicating which parts the model should focus on.
Sum to 1.0 across the attended sequence.

### **Context Vector (Attention Context)**
Weighted sum of input representations based on attention weights.
Summarizes relevant information from the input sequence.

### **Cross-Attention**
Attention mechanism between two different sequences (e.g., question attending to document).

### **Self-Attention**
Attention mechanism where a sequence attends to itself, allowing each position to consider all other positions in the same sequence.

## **Neural Network Fundamentals**

### **Embedding Layer**
Neural network layer that maps discrete tokens (words) to dense vector representations, capturing semantic relationships.

### **Encoder-Decoder Architecture**
Neural network design where an encoder processes input into a fixed representation, and a decoder generates output from that representation.

### **Hidden State**
Internal memory of recurrent neural networks that maintains information from previous time steps.

### **LSTM (Long Short-Term Memory)**
Type of RNN designed to handle long sequences by using gates to control information flow, solving the vanishing gradient problem.

### **Softmax Function**
Function that converts a vector of real numbers into a probability distribution where all values sum to 1.0.

## **Training and Optimization**

### **Adam Optimizer**
Adaptive optimization algorithm that computes individual learning rates for different parameters based on estimates of gradient moments.

### **Cross-Entropy Loss**
Loss function commonly used for classification tasks, measuring the difference between predicted and true probability distributions.

### **Dropout**
Regularization technique that randomly sets some neurons to zero during training to prevent overfitting.

### **Gradient Clipping**
Technique to prevent exploding gradients by limiting the magnitude of gradients during backpropagation.

### **Learning Rate Scheduling**
Strategy for adjusting the learning rate during training, often decreasing it over time for better convergence.

## **Evaluation and Metrics**

### **Ablation Study**
Systematic removal of model components to understand their individual contributions to overall performance.

### **Baseline Model**
Simple model used as a reference point for comparison, often representing the minimal reasonable approach to a problem.

### **Development Set (Dev Set)**
Subset of data used for hyperparameter tuning and model validation during development, separate from test set.

### **Ensemble Model**
Combination of multiple models whose predictions are aggregated (e.g., averaged) to achieve better performance than individual models.

### **Leaderboard**
Public ranking of model performances on a benchmark dataset, allowing comparison between different approaches.

## **Text Processing**

### **GloVe (Global Vectors)**
Pre-trained word embeddings that capture semantic relationships between words based on global word co-occurrence statistics.

### **Out-of-Vocabulary (OOV)**
Words that appear in test data but not in the training vocabulary, typically handled with special `<UNK>` tokens.

### **Stanford CoreNLP**
Natural language processing toolkit providing tokenization, POS tagging, named entity recognition, and other text processing capabilities.

### **Tokenization**
Process of splitting text into individual units (tokens), typically words or subwords, for neural network processing.

### **Vocabulary**
Set of all unique tokens that a model can process, often including special tokens like `<UNK>` for unknown words.

## **Advanced Concepts**

### **Beam Search**
Decoding strategy that maintains multiple candidate sequences during generation, exploring more possibilities than greedy search.

### **Highway Networks**
Architecture allowing information to flow directly through networks via learned gating mechanisms, enabling training of very deep networks.

### **Iterative Conditional Modes (ICM)**
Optimization algorithm that iteratively updates variables while keeping others fixed, inspiring DCN's dynamic decoder approach.

### **Maxout Networks**
Neural networks using the max operation instead of traditional activation functions, providing model capacity without adding parameters.

### **Residual Connections**
Skip connections that add the input of a layer to its output, helping with gradient flow in deep networks.

## **Implementation Details**

### **Batch Processing**
Processing multiple examples simultaneously for computational efficiency, requiring padding to handle variable-length sequences.

### **Checkpointing**
Saving model parameters periodically during training to enable recovery from failures and model evaluation.

### **Masking**
Technique to ignore padded positions in loss computation and attention mechanisms, ensuring they don't affect model predictions.

### **Mixed Precision Training**
Using both 16-bit and 32-bit floating point numbers to speed up training while maintaining numerical stability.

### **Sequence Padding**
Adding special padding tokens to make all sequences in a batch the same length for efficient tensor operations.

## **Related Models and Techniques**

### **BERT (Bidirectional Encoder Representations from Transformers)**
Pre-trained language model that can be fine-tuned for question answering and other NLP tasks.

### **BiDAF (Bidirectional Attention Flow)**
QA model using bidirectional attention between question and context, influential predecessor to DCN.

### **Match-LSTM**
Sequence matching model using LSTM to process question-document pairs with attention mechanisms.

### **Pointer Networks**
Neural architecture that learns to output positions in the input sequence rather than generating from a fixed vocabulary.

### **Transformer**
Attention-based architecture that revolutionized NLP, using self-attention instead of recurrent connections.

---