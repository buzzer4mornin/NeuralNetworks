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
size_of_data = 10000
train_data = [reformat_to_window(shuffle_letters(random.choice(word_list), random.randint(1, 3) * random.randint(0, 1))) \
              for _ in range(0, size_of_data)]

# ========================================= Encode & Decode output ====================================================

nn_output = np.zeros(len(word_list))

"""In order to train the model need a supervised train_data. Let's encode the output as a vector of binaries where 1 
in i ith position if the model recognize the i ith word of the list. And declare as a non-sense if its recognize two 
different words. """

recognize = lambda w: np.array([1 if w == w_f else 0 for w_f in word_list_formated])
train_y = [recognize(w) for w in train_data]
print(sum([sum(x) for x in train_y])/len(train_data)) # proportion of correct words after shuffle

