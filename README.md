# DL- Developing a Deep Learning Model for NER using LSTM

## AIM
To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement and Dataset
The goal of this experiment is to build and train a deep learning model, specifically a Bi-directional Long Short-Term Memory (BiLSTM) network, to perform Named Entity Recognition (NER). Given a sentence, the model should be able to identify and classify named entities such as geographical locations, organizations, persons, and time indicators within the text.

<img width="403" height="261" alt="Screenshot 2026-03-10 111548" src="https://github.com/user-attachments/assets/613c38e2-5b75-43ce-82e5-9645f9111f1d" />

## DESIGN STEPS
### STEP 1: 

Data Loading and Preprocessing:

Load the ner_dataset.csv containing words, their Part-of-Speech (POS) tags, and Named Entity (NE) tags.
Extract unique words and tags to create dictionaries (word2idx, tag2idx) for mapping words and tags to numerical indices.
Group words and their corresponding tags into full sentences.
Encode each word in a sentence and its tag into their respective numerical indices.


### STEP 2: 

Data Padding and Splitting:

Determine a maximum sentence length (max_len) and pad shorter sentences with a special ENDPAD token (and 'O' tag for labels) to ensure all input sequences have uniform length.
Split the padded dataset into training and testing sets to evaluate the model's generalization capabilities.

### STEP 3: 

Dataset and DataLoader Creation:

Implement a custom PyTorch Dataset class to manage the input features (word indices) and labels (tag indices).
Create DataLoader instances for both the training and testing sets to efficiently load data in mini-batches during model training and evaluation.

### STEP 4: 
Model Definition and Initialization:

Define the BiLSTM model architecture (BiLSTMTagger) using PyTorch's nn.Module. This model will consist of:
An nn.Embedding layer to convert word indices into dense vector representations.
A nn.Dropout layer for regularization.
An nn.LSTM layer configured as bidirectional to capture context from both past and future words.
A final nn.Linear layer to project the LSTM outputs to the size of the tag set.
Initialize the model, define the CrossEntropyLoss as the loss function, and select the Adam optimizer for updating model weights.


### STEP 5: 
Model Training and Evaluation:

Implement a training loop (train_model) that iterates for a specified number of epochs, performs forward and backward passes, calculates loss, and updates model parameters.
Implement an evaluation function (evaluate_model) to assess the model's performance on unseen test data, focusing on metrics relevant to NER (though the current notebook only prints classification report, a full metric calculation for NE entities would be here).







## PROGRAM

### Name: Sanchita Sandeep

### Register Number: 212224240142

```python
class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size,tagset_size, embedding_dim=50, hidden_dim=100):
      super(BiLSTMTagger, self).__init__()
      self.embedding = nn.Embedding(vocab_size, embedding_dim)
      self.dropout = nn.Dropout(0.1)
      self.lstm = nn.LSTM(embedding_dim,hidden_dim, batch_first=True, bidirectional=True)
      self.fc = nn.Linear(hidden_dim*2,tagset_size)
     
    def forward(self,x):
      x = self.embedding(x)
      x = self.dropout(x)
      x, _ = self.lstm(x)
      return self.fc(x)
from torch.nn.modules import loss
from functools import total_ordering
# Training and Evaluation Functions
def train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs=3):
    # Include the training and evaluation functions
    train_losses, val_losses = [], []
    for epoch in range(epochs):
      model.train()
      total_loss=0
      for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = loss_fn(outputs.view(-1, len(tag2idx)), labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
      train_losses.append(total_loss)  

      model.eval()
      val_loss = 0
      with torch.no_grad():
        for batch in test_loader:
          input_ids = batch["input_ids"].to(device)
          labels = batch["labels"].to(device)
          outputs = model(input_ids)
          loss = loss_fn(outputs.view(-1, len(tag2idx)), labels.view(-1))
          val_loss += loss.item()
      val_losses.append(val_loss)
      print(f"Epoch {epoch+1}: Train loss = {total_loss:.4f}, Val loss = {val_loss:.4f}")
    return train_losses, val_losses



```

### OUTPUT

## Loss Vs Epoch Plot
<img width="784" height="624" alt="Screenshot 2026-03-10 111314" src="https://github.com/user-attachments/assets/f661c668-774e-44de-bf63-763ffe20bcf2" />



### Sample Text Prediction

<img width="417" height="523" alt="Screenshot 2026-03-10 111324" src="https://github.com/user-attachments/assets/5a8c9f87-12bd-40d2-92f7-d39bf4c59692" />



## RESULT
The model predicts the resulis sucessfully
