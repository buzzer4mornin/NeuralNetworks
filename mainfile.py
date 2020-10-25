import random
import numpy as np

"""
Generating the training data
Taking a random list of 20 words with this website : https://www.randomlists.com/random-words?dup=false&qty=20"""
word_list = ['nice', 'toothpaste', 'halting', 'deafening', 'structure', 'soak', 'omniscient', 'found', 'authority',
             'crabby', 'blush', 'dreary', 'childlike', 'nippy', 'roll', 'highfalutin', 'wing', 'hole', 'nonchalant',
             'plant']

window_length = max([len(w) for w in word_list]) + 1
reformat_to_window = lambda w: w + ' ' * (window_length - len(w))
word_list_formated = [reformat_to_window(w) for w in word_list]

"""
Generating the appropriate train data : 
- Balanced 
- Negative closed to positive """


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
size_of_data = 30000
train_data = [reformat_to_window(shuffle_letters(random.choice(word_list), random.randint(1, 3) * random.randint(0, 1))) \
              for _ in range(0, size_of_data)]
# ========================================= Encode & Decode output ====================================================
nn_output = np.zeros(len(word_list))
"""In order to train the model need a supervised train_data. Let's encode the output as a vector of binaries where 1 
in i ith position if the model recognize the i ith word of the list. And declare as a non-sense if its recognize two 
different words. """

recognize = lambda w: np.array([1 if w == w_f else 0 for w_f in word_list_formated])
train_y = [recognize(w) for w in train_data]
# print(sum([sum(x) for x in train_y]) / len(train_data))  # proportion of correct words after shuffle

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
nn_hidden_layer_1 = np.zeros(20)
nn_hidden_layer_2 = np.zeros(10)

# print(nn_encode.size, nn_hidden_layer_1.size, nn_output.size)
""" Why 60-10-20? 
Because in Input, we encode each input as length of 12 -> "blink       " 
and for each character in input, we encode it as binary of length 5, so 5*12 = 60 input neurons
in Output we get 20 neurons and each neuron is 0/1. It is map decision of our Recognition Word list created 
at very beginning of task - word_list. Each binary output means whether NN could recognize given input as any member 
of word_list
"""

# Link the layer by neurons:
sigmoid = lambda x: 1 / (1 + np.exp(-x))
sigmoid_ = lambda y: y * (1 - y)


class NeuralNetwork:
    def __init__(self, input_layer=nn_encode, l_hidden_layers=[nn_hidden_layer_1], output_layer=nn_output):
        self.layers = np.array([input_layer, *l_hidden_layers, output_layer])
        L = []
        for i in range(0, self.layers.size - 1):  # !!!! L for weigth corespond to L-1 for Layers
            weight = np.random.rand(self.layers[i].size + 1, self.layers[i + 1].size)
            weight[-1] = -1 * weight[-1]  # Change the signe of the biase
            L.append(weight)  # self.weights[L][k][l] : L : Layer  ; k neurone ; l edge
        self.weights = np.array(L)
        

    def input_word(self, word):
        self.layers[0] = encode_window(word)

    def input_decode(self):
        return (decode_window(self.layers[0]))

    def output(self):
        return (self.layers[-1])

    def decode_output(self):
        return (decode_output((self.layers[-1] >= 0.5)) * 1)

    def feedfoward(self):
        for L in range(0, self.layers.size - 1):
            self.layers[L + 1] = sigmoid(np.dot(np.array([*self.layers[L], 1]), self.weights[L]))

    def backpropagation(self, desiered_output, eta=0.1):
        error_vector = self.output() - desiered_output
        cost = np.sum(error_vector * error_vector)
        for L in range(0, self.layers.size - 1):
            L = self.layers.size - 2 - L  # browse the list backwards
            delta = sigmoid_(self.layers[L + 1]) * error_vector  #
            mat_delta = np.matmul(np.array([*self.layers[L], 1]).reshape(self.layers[L].size + 1, 1),
                                  delta.reshape(1, delta.size))  ## Adding 1 for
            self.weights[L] = self.weights[L] - eta * 2 * mat_delta  ## Actualize the weight
            error_vector = np.matmul(self.weights[L][0:-1], error_vector.T)  ## Propagate the error (without biases)
        return cost


nn = NeuralNetwork(l_hidden_layers=[nn_hidden_layer_1])
# print(nn.weights[0][0][0])


# Split Train and Test data
# test_data = train_data[0:2000]
# train_data = train_data[2000:10000]
# train_y_train = train_y[2000:10000]
# train_y_test = train_y[0:2000]


p = 0
for i, w in enumerate(train_data):
    nn.input_word(w)
    nn.feedfoward()
    cost = nn.backpropagation(desiered_output=train_y[i], eta=0.2)
    #print(cost)
    '''if p >= 100:
        print(cost)
        p = 0
    p = p + 1'''

counts = 0
for word in word_list:
    nn.input_word(reformat_to_window(word))
    nn.feedfoward()
    if nn.decode_output() == word:
        counts += 1

print(counts)


"""t = np.ndarray(1).reshape(1, -1)
for i in train_y:
    if sum(i) == 1:
        t = np.concatenate((t, [[list(i).index(1)]]), axis=0)
    else:
        t = np.concatenate((t, [[20]]), axis=0)
t = t[1:]
#tensor_y = torch.Tensor(np.array(t))
"""