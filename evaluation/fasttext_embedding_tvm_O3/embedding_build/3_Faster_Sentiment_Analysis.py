#!/usr/bin/env python
# coding: utf-8

# # 3 - Faster Sentiment Analysis
# 
# In the previous notebook we managed to achieve a decent test accuracy of ~84% using all of the common techniques used for sentiment analysis. In this notebook, we'll implement a model that gets comparable results whilst training significantly faster and using around half of the parameters. More specifically, we'll be implementing the "FastText" model from the paper [Bag of Tricks for Efficient Text Classification](https://arxiv.org/abs/1607.01759).

# ## Preparing Data
# 
# One of the key concepts in the FastText paper is that they calculate the n-grams of an input sentence and append them to the end of a sentence. Here, we'll use bi-grams. Briefly, a bi-gram is a pair of words/tokens that appear consecutively within a sentence. 
# 
# For example, in the sentence "how are you ?", the bi-grams are: "how are", "are you" and "you ?".
# 
# The `generate_bigrams` function takes a sentence that has already been tokenized, calculates the bi-grams and appends them to the end of the tokenized list.

# In[1]:


def generate_bigrams(x):
    n_grams = set(zip(*[x[i:] for i in range(2)]))
    for n_gram in n_grams:
        x.append(' '.join(n_gram))
    return x


# As an example:

# In[2]:


generate_bigrams(['This', 'film', 'is', 'terrible'])


# TorchText `Field`s have a `preprocessing` argument. A function passed here will be applied to a sentence after it has been tokenized (transformed from a string into a list of tokens), but before it has been numericalized (transformed from a list of tokens to a list of indexes). This is where we'll pass our `generate_bigrams` function.
# 
# As we aren't using an RNN we can't use packed padded sequences, thus we do not need to set `include_lengths = True`.

# In[3]:


import torch
from torchtext.legacy import data
from torchtext.legacy import datasets

SEED = 1234

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

TEXT = data.Field(tokenize = 'spacy',
                  tokenizer_language = 'en_core_web_sm',
                  preprocessing = generate_bigrams)

LABEL = data.LabelField(dtype = torch.float)


# As before, we load the IMDb dataset and create the splits.

# In[4]:


import random
device = 'cpu'
import spacy
nlp = spacy.load('en_core_web_sm')

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

train_data, valid_data = train_data.split(random_state = random.seed(SEED))


# Build the vocab and load the pre-trained word embeddings.

# In[5]:


MAX_VOCAB_SIZE = 25_000

TEXT.build_vocab(train_data, 
                 max_size = MAX_VOCAB_SIZE, 
                 vectors = "glove.6B.100d", 
                 unk_init = torch.Tensor.normal_)

LABEL.build_vocab(train_data)
'''
print(type(train_data))
print(train_data)
min_size = 10000
for t_d in train_data:
    #print(t_d.text)
    #print(vars(t_d))
    #print(type(t_d))
    #exit(0)
    sentence = ' '.join(t_d.text)
    tokenized = generate_bigrams([tok.text for tok in nlp.tokenizer(sentence)])
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    #print(tensor)
    #print(tensor.size())
    if tensor.size()[0] <= 7:
        print(sentence)
        exit(0)
    elif tensor.size()[0] < min_size:
        min_size = tensor.size()[0]

print('no one found')    
print('min_size {}'.format(min_size))
exit(0)
'''
# And create the iterators.

# In[6]:


BATCH_SIZE = 64

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size = BATCH_SIZE, 
    device = device)

'''
# ========
# To get some data
print('To Get Some Data')
for batch in valid_iterator:
    print(type(batch))
    print(batch.text)
    print(type(batch.text))
    exit(0)

# ========
'''
# ## Build the Model
# 
# This model has far fewer parameters than the previous model as it only has 2 layers that have any parameters, the embedding layer and the linear layer. There is no RNN component in sight!
# 
# Instead, it first calculates the word embedding for each word using the `Embedding` layer (blue), then calculates the average of all of the word embeddings (pink) and feeds this through the `Linear` layer (silver), and that's it!
# 
# ![](assets/sentiment8.png)
# 
# We implement the averaging with the `avg_pool2d` (average pool 2-dimensions) function. Initially, you may think using a 2-dimensional pooling seems strange, surely our sentences are 1-dimensional, not 2-dimensional? However, you can think of the word embeddings as a 2-dimensional grid, where the words are along one axis and the dimensions of the word embeddings are along the other. The image below is an example sentence after being converted into 5-dimensional word embeddings, with the words along the vertical axis and the embeddings along the horizontal axis. Each element in this [4x5] tensor is represented by a green block.
# 
# ![](assets/sentiment9.png)
# 
# The `avg_pool2d` uses a filter of size `embedded.shape[1]` (i.e. the length of the sentence) by 1. This is shown in pink in the image below.
# 
# ![](assets/sentiment10.png)
# 
# We calculate the average value of all elements covered by the filter, then the filter then slides to the right, calculating the average over the next column of embedding values for each word in the sentence. 
# 
# ![](assets/sentiment11.png)
# 
# Each filter position gives us a single value, the average of all covered elements. After the filter has covered all embedding dimensions we get a [1x5] tensor. This tensor is then passed through the linear layer to produce our prediction.

# In[7]:


import torch.nn as nn
import torch.nn.functional as F

class FastText(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, pad_idx):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        self.fc = nn.Linear(embedding_dim, output_dim)
        
    def forward(self, text):
        
        #text = [sent len, batch size]
        
        embedded = self.embedding(text)
                
        #embedded = [sent len, batch size, emb dim]
        
        embedded = embedded.permute(1, 0, 2)
        
        #embedded = [batch size, sent len, emb dim]
        
        pooled_shape = [int(embedded.shape[1]), 1]
        embedded = embedded.reshape(embedded.shape[0], 1, embedded.shape[1], embedded.shape[2])  # modified
        
        pooled = F.avg_pool2d(embedded, pooled_shape) 
        # pooled = F.avg_pool2d(embedded, (7, 1))  # modified
        
        pooled = pooled.squeeze(1)  # modified
        
        #pooled = [batch size, embedding_dim]
                
        return self.fc(pooled)


# As previously, we'll create an instance of our `FastText` class.

# In[8]:


INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
OUTPUT_DIM = 1
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
print('PAD_IDX', PAD_IDX)
print(INPUT_DIM)
exit(0)
model = FastText(INPUT_DIM, EMBEDDING_DIM, OUTPUT_DIM, PAD_IDX)


# Looking at the number of parameters in our model, we see we have about the same as the standard RNN from the first notebook and half the parameters of the previous model.

# In[9]:


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')


# And copy the pre-trained vectors to our embedding layer.

# In[10]:


pretrained_embeddings = TEXT.vocab.vectors

model.embedding.weight.data.copy_(pretrained_embeddings)


# Not forgetting to zero the initial weights of our unknown and padding tokens.

# In[11]:


UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)


# ## Train the Model

# Training the model is the exact same as last time.
# 
# We initialize our optimizer...

# In[12]:


import torch.optim as optim

optimizer = optim.Adam(model.parameters())


# We define the criterion and place the model and criterion on the GPU (if available)...

# In[13]:


criterion = nn.BCEWithLogitsLoss()

model = model.to(device)
criterion = criterion.to(device)


# We implement the function to calculate accuracy...

# In[14]:


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc


# We define a function for training our model...
# 
# **Note**: we are no longer using dropout so we do not need to use `model.train()`, but as mentioned in the 1st notebook, it is good practice to use it.

# In[15]:


def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
        
        predictions = model(batch.text).squeeze(1)
        predictions = predictions.squeeze(1)
        # print(predictions.size())
        
        loss = criterion(predictions, batch.label)
        
        acc = binary_accuracy(predictions, batch.label)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# We define a function for testing our model...
# 
# **Note**: again, we leave `model.eval()` even though we do not use dropout.

# In[16]:


def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            predictions = model(batch.text).squeeze(1)
            predictions = predictions.squeeze(1)
            
            loss = criterion(predictions, batch.label)
            
            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# As before, we'll implement a useful function to tell us how long an epoch takes.

# In[17]:


import time

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# Finally, we train our model.

# In[18]:


N_EPOCHS = 5

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()
    
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut3-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')


# ...and get the test accuracy!
# 
# The results are comparable to the results in the last notebook, but training takes considerably less time!

# In[19]:


model.load_state_dict(torch.load('tut3-model.pt'))

test_loss, test_acc = evaluate(model, test_iterator, criterion)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')


# ## User Input
# 
# And as before, we can test on any input the user provides making sure to generate bigrams from our tokenized sentence.

# In[20]:


import spacy
nlp = spacy.load('en_core_web_sm')

def predict_sentiment(model, sentence):
    model.eval()
    tokenized = generate_bigrams([tok.text for tok in nlp.tokenizer(sentence)])
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    print(tensor)
    print(tensor.size())
    tensor = tensor.unsqueeze(1)
    print(tensor)
    print(tensor.size())
    prediction = torch.sigmoid(model(tensor))
    return prediction.item()


# An example negative review...

# In[21]:


print(predict_sentiment(model, "This film is terrible"))


# An example positive review...

# In[22]:


print(predict_sentiment(model, "This film is great"))

dummy_input = torch.randint(0, 10000, (7, 1))
# torch.onnx.export(model, dummy_input, "embedding_2.onnx", opset_version=12,)
