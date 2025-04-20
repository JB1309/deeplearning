# deeplearning
This repository contains code for training a sequence-to-sequence (Seq2Seq) model for transliteration using Recurrent Neural Networks (RNNs), specifically LSTM cells, implemented in PyTorch. The model is designed for transliteration tasks, where a sequence in one language is converted into a corresponding sequence in another language.

Requirements
To run the code, you need the following libraries:

Pandas for data processing

Install the required libraries using pip:

pip install torch pandas
File Structure
train_model.py: Main training script that trains the Seq2Seq model.

hi.translit.sampled.train.tsv: The training dataset (replace with your own data if needed).

hi.translit.sampled.dev.tsv: The development dataset (replace with your own data if needed).

README.md: This file.

Dataset
The dataset consists of two columns:

src: Source language text (e.g., Hindi transliteration).

tgt: Target language text (e.g., Romanized form).

The dataset should be in TSV (tab-separated values) format, where each row corresponds to a pair of source and target language text.

Example:

css
Copy
Edit
src	tgt
अ	a
आप	āp
Model Architecture
1. Encoder
The encoder uses an embedding layer followed by an RNN layer (LSTM). The input sequence is embedded and passed through the RNN to obtain hidden states.

2. Decoder
The decoder is also an RNN (LSTM). It takes the output from the encoder (hidden states) and generates the target sequence. The decoder uses teacher forcing during training, which means it uses the actual previous target token as input for the next time step with a certain probability (controlled by TEACHER_FORCING_RATIO).

3. Seq2Seq Model
The Seq2Seq model connects the encoder and decoder, and handles the generation of the target sequence based on the source sequence.

Hyperparameters
EMBED_DIM: Dimension of the embedding vectors.

HIDDEN_DIM: Number of units in the hidden layer of the LSTM.

NUM_LAYERS: Number of LSTM layers.

RNN_TYPE: Type of RNN to use (LSTM or GRU).

BATCH_SIZE: Batch size for training.

EPOCHS: Number of training epochs.

TEACHER_FORCING_RATIO: Probability of using teacher forcing during training.

Training the Model
Dataset Format
Make sure you have the training (hi.translit.sampled.train.tsv) and validation (hi.translit.sampled.dev.tsv) files in the correct format. If you have your own dataset, update the train_file and dev_file paths in the code.

Steps to Train:
Place your training (train.tsv) and validation (dev.tsv) dataset files in the correct paths.

Run the training script:

bash
Copy
Edit
python train_model.py
Training Process:
Data Loading: The dataset is loaded using PyTorch's DataLoader. The dataset is tokenized and padded to handle variable-length sequences.

Model Training: The model is trained for the specified number of epochs using the Adam optimizer. During each epoch, the loss is calculated using the cross-entropy loss function.

Evaluation: After each epoch, the model is evaluated on the validation set. The accuracy is calculated based on the number of correctly predicted transliterations.

Teacher Forcing: During training, teacher forcing is applied with the specified ratio (TEACHER_FORCING_RATIO).

Evaluation
The evaluation step runs after each epoch and calculates the accuracy by comparing the predicted transliterations with the ground truth. Additionally, the first five examples (source, prediction, target) are printed for inspection.

Sample Output
During training, the script will print the following information:

yaml
Copy
Edit
Epoch 1/10
Training Loss: 1.2345
Validation Accuracy: 0.8534
Source: आप | Prediction: aap | Target: aap
Source: नमस्ते | Prediction: namaste | Target: namaste
Source: धन्यवाद | Prediction: dhanyavaad | Target: dhanyavaad
...
Model Inference
After training, the model can be used for inference by passing a source text sequence through the encoder and decoder.

python
Copy
Edit
# Example inference:
encoder.eval()
decoder.eval()

input_text = "आप"
input_tensor = torch.tensor([train_dataset.encode(input_text, train_dataset.src_vocab)]).to(device)
hidden = encoder(input_tensor)
output, _ = decoder(input_tensor, hidden)
output_text = train_dataset.decode(output.argmax(-1).cpu().tolist(), train_dataset.tgt_vocab)
print(f"Transliteration: {output_text}")
Notes
Teacher Forcing: The TEACHER_FORCING_RATIO controls how much teacher forcing is applied during training. Setting it to 1 means always using the actual target for the next time step, and setting it to 0 means using the model's previous prediction.

Padding: The sequences are padded using the <pad> token to handle variable-length input and output sequences.
