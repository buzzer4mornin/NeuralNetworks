import random
import numpy as np
import lzma
import pickle
from sklearn.neural_network import MLPClassifier

# ======================================================================================================================
"""Generating the training data
Taking a random list of 20 words with this website : https://www.randomlists.com/random-words?dup=false&qty=20
"""
word_list = ['nice', 'toothpaste', 'halting', 'deafening', 'structure', 'soak', 'omniscient', 'found', 'authority',
             'crabby', 'blush', 'dreary', 'childlike', 'nippy', 'roll', 'highfalutin', 'wing', 'hole', 'nonchalant',
             'plant']

window_length = max([len(w) for w in word_list]) + 1
reformat_to_window = lambda w: w + ' ' * (window_length - len(w))
word_list_formated = [reformat_to_window(w) for w in word_list]


# ======================================================================================================================
def shuffle_letters(word, shuffle_letterse=5):
    """ Recursive function that exchange two letters at each iteration (shuffle)"""
    if type(word) is not list:
        word = list(word)
    if shuffle_letterse == 0:
        return ''.join(word)  # word==shuffle_letters(word,shuffle=0) -> TRUE
    i, j = random.randint(0, len(word) - 1), random.randint(0, len(word) - 1)
    word[i], word[j] = word[j], word[i]
    return shuffle_letters(word, shuffle_letterse - 1)


# ======================================================================================================================
size_of_data = 10000
train_data = [reformat_to_window(shuffle_letters(random.choice(word_list), random.randint(1, 3) * random.randint(0, 1))) \
              for _ in range(0, size_of_data)]

# ======================================================================================================================
"""In order to train the model need a supervised train_data. Let's encode the output as a vector of binaries where 1 
in i ith position if the model recognize the i ith word of the list. And declare as a non-sense if its recognize two 
different words. 
"""
nn_output = np.zeros(len(word_list))

recognize = lambda w: np.array([1 if w == w_f else 0 for w_f in word_list_formated])
train_y = [recognize(w) for w in train_data]

# Print proportion of correct words after shuffle
# print(sum([sum(x) for x in train_y]) / len(train_data))

decode_output = lambda o: 'Not recognize' if np.sum(o) >= 2 or np.sum(o) == 0 else word_list[o.tolist().index(1)]
nn_output = train_y[1]
# print(nn_output)
# print(decode_output(nn_output))

# =====================================================================================================================
# ============================================== INPUT ENCODING =======================================================
# =====================================================================================================================
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


nn_encode = [encode_window(word=train_data[i]) for i in range(len(train_data))]
# print(nn_encode)
# print(decode_window(nn_encode))

# =====================================================================================================================
# ============================================ NEURAL NETWORK =========================================================
# =====================================================================================================================
network = MLPClassifier(hidden_layer_sizes=(10, 10), activation='relu', solver='adam', alpha=0.0001,
                        batch_size=300, learning_rate='constant', learning_rate_init=0.001, max_iter=200,
                        shuffle=True, random_state=42, tol=1e-5, verbose=True, early_stopping=False)

model = network.fit(nn_encode, train_y)

with lzma.open("diacritization.model", "wb") as model_file:
    pickle.dump(model, model_file)
with lzma.open("diacritization.model", "rb") as model_file:
       model = pickle.load(model_file)

#for word in word_list:
#    print(model.predict(encode_window(reformat_to_window(word)).reshape(1, -1)))

test_size = 100000
mytest = [shuffle_letters(random.choice(word_list), random.randint(1, 5) * random.randint(0, 1)) \
          for _ in range(0, test_size)]


c = 0
for word in mytest:
    if word not in word_list:
        pred = decode_output(model.predict(encode_window(reformat_to_window(word)).reshape(1, -1))[0])
        print(word, " ====== ", pred)
        if pred != "Not recognize":
            c += 1
print(c)
