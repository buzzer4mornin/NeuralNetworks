import random
import numpy as np
import pandas as pd
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import optim

"""
Generating the training data
Taking a random list of 20 words with this website : https://www.randomlists.com/random-words?dup=false&qty=20"""
word_list = ['nice', 'toothpaste', 'halting', 'deafening', 'structure', 'soak', 'omniscient', 'found', 'authority',
             'crabby', 'blush', 'dreary', 'childlike', 'nippy', 'roll', 'highfalutin', 'wing', 'hole', 'nonchalant',
             'plant']

window_length = max([len(w) for w in word_list]) + 1
reformat_to_window = lambda w: w + ' ' * (window_length - len(w))
word_list_formated = [reformat_to_window(w) for w in word_list]


def shuffle_letters(word, shuffle=5):
    """ Recursive function that exchange two letters at each iteration (shuffle)"""
    if type(word) is not list:
        word = list(word)
    if shuffle == 0:
        return ''.join(word)  # word==shuffle_letters(word,shuffle=0) -> TRUE
    i, j = random.randint(0, len(word) - 1), random.randint(0, len(word) - 1)
    word[i], word[j] = word[j], word[i]
    return shuffle_letters(word, shuffle - 1)


"""Generate training data"""
size_of_data = 3000
train_data = [reformat_to_window(shuffle_letters(random.choice(word_list), random.randint(1, 3) * random.randint(0, 1))) \
              for _ in range(0, size_of_data)]
# ========================================= Encode & Decode output ====================================================
nn_output = np.zeros(len(word_list))
"""In order to train the model need a supervised train_data. Let's encode the output as a vector of binaries where 1 
in i ith position if the model recognize the i ith word of the list. And declare as a non-sense if its recognize two 
different words. """

recognize = lambda w: np.array([1 if w == w_f else 0 for w_f in word_list_formated])
train_y = [recognize(w) for w in train_data]

#print(sum([sum(x) for x in train_y]) / len(train_y))  # proportion of correct words after shuffle

decode_output = lambda o: 'Not recognize' if np.sum(o) >= 2 or np.sum(o) == 0 else word_list[o.tolist().index(1)]
# nn_output = train_y[2]
# print(nn_output)
# print(decode_output(nn_output))

# ========================================= Encode & Decode input ====================================================
"""
Encode the input look at this https://stackoverflow.com/questions/7396849/convert-binary-to-ascii-and-vice-versa
"""

# Here is a little demonstration that we need 5 binaries to distinguish the 26 minus letter
alpha = 'abcdefghijklmnopqrstuvwxyz'
binaries = [bin(ord(letter_min))[-5:] for letter_min in list(alpha)]
# print(len(binaries), len(list(set(binaries)))) # suppress the doublon in the second list

nn_encode = np.zeros(window_length * 5)


def encode_window(word=word_list_formated[0]):
    # print(word)
    global nn_encode
    word_binaries = [bin(ord(l))[-5:] for l in list(word)]
    word_binaries = ''.join(word_binaries)
    word_binaries = [int(b) for b in list(word_binaries)]
    # print(word_binaries)
    return np.array(word_binaries)


def decode_window(input_binaries_list):
    word_binaries = input_binaries_list.tolist()
    word_binaries = ''.join([str(b) for b in input_binaries_list])
    word_binaries = ['011' + word_binaries[i:i + 5] for i in range(0, len(word_binaries), 5)]
    # print(word_binaries)
    return ''.join([chr(int(x, 2)) for x in word_binaries])


nn_encode = encode_window(word=word_list_formated[0])
decode_window(nn_encode)

# ============================================== Neural Network =======================================================

# print(train_data[5])   # word itself
# print(encode_window(word=train_data[5])) # Input binaries
# print(train_y[5])   # Output desired binaries

input_size = 60
hidden_layers = [10, 10]
output_size = 20

c = encode_window(word=train_data[0]).reshape(1, -1)
for i in range(1, len(train_data)):
    a = encode_window(word=train_data[i]).reshape(1, -1)
    c = np.concatenate((c, a), axis=0)

test_x = c[2800:3000]
test_y = np.array(train_y)[2800:3000]
train_x = c[0:2800]
train_y = np.array(train_y)[0:2800]

tensor_x = torch.Tensor(train_x)
tensor_y = torch.Tensor(train_y)
#print(tensor_x.shape, tensor_y.shape)

tensor_x_test = torch.Tensor(test_x)
tensor_y_test = torch.Tensor(test_y)
#print(tensor_x_test.shape, tensor_y_test.shape)


model = nn.Sequential(
    nn.Linear(input_size, hidden_layers[0]),
    nn.ReLU(),
    nn.Linear(hidden_layers[0], hidden_layers[1]),
    nn.ReLU(),
    nn.Linear(hidden_layers[1], output_size),
    #nn.Softmax(dim=1)
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)


def softmax(x): return x.exp() / (x.exp().sum(-1)).unsqueeze(-1)
def nl(input, target): return -input[range(target.shape[0]), target].log().mean()

for e in range(4):
    running_loss = 0
    for images, labels in zip(tensor_x, tensor_y):

        output = model(images)
        # =========================================================
        #print(output.shape, labels.shape)
        #output = output.view(1, -1)
        #labels = labels.view(1, -1)
        #probs = nn.LogSoftmax(dim=0)
        #softmax_output = probs(output)
        #loss = criterion(output, labels)
        # =========================================================
        output = softmax(output)
        output = output.view(1, -1)
        labels = labels.view(1, -1)
        output = output.type(torch.FloatTensor)
        labels = labels.type(torch.LongTensor)
        loss = nl(output, labels)

        # set gradient to zero
        optimizer.zero_grad()
        # backward propagation
        loss.backward()

        # update the gradient to new gradients
        optimizer.step()
        running_loss += loss.item()
    print("Training loss: ", (running_loss / 2800))






def check_acc(tensor_x_test,tensor_y_test, model):
    model.eval()
    with torch.no_grad():
        for images, labels in zip(tensor_x_test, tensor_y_test):
            output = model(images)
            # output = softmax(output)
            output = output.view(1, -1)
            print(output)

check_acc(tensor_x_test, tensor_y_test, model)