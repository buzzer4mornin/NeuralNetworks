import csv
import random
import numpy as np
import lzma
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error

# ======================================================================================================================
# =========================================== GENERATE POSITIVE WORDS ==================================================

# Taking a random list of 7 words with this website : https://www.randomlists.com/random-words?dup=false&qty=7
word_list = ['nice', 'toothpaste', 'highfalutin', 'childlike', 'authority', 'halting', 'found']

window_length = max([len(w) for w in word_list]) + 1
reformat_to_window = lambda w: w + ' ' * (window_length - len(w))
word_list_formated = [reformat_to_window(w) for w in word_list]

with open('./positive_data.txt', 'w') as f:
    f.write("WORD" + 15 * " " + "OUTPUT" + "\n")
    for i in range(len(word_list_formated)):
        out = np.zeros(7)
        out[i] = 1
        f.write(word_list_formated[i])
        f.write(str(out))
        f.write("\n")


# =====================================================================================================================
# ============================================ GENERATE TRAIN DATA =====================================================

def shuffle_letters(word, shuffle_letterse=5):
    """ Recursive function that exchange two letters at each iteration (shuffle)"""
    if type(word) is not list:
        word = list(word)
    if shuffle_letterse == 0:
        return ''.join(word)  # word==shuffle_letters(word,shuffle=0) -> TRUE
    i, j = random.randint(0, len(word) - 1), random.randint(0, len(word) - 1)
    word[i], word[j] = word[j], word[i]
    return shuffle_letters(word, shuffle_letterse - 1)


# Generate train data with both positive and negative words
size_of_data = 10000
train_data = [reformat_to_window(shuffle_letters(random.choice(word_list), random.randint(1, 3) * random.randint(0, 1))) \
              for _ in range(0, size_of_data)]

# In order to train the model, we also need train_target. So, lets encode the output as a vector of binaries where "1"
# will be in ith position if the model recognize input word as ith word from original positive list of 7 words """
recognize = lambda w: np.array([1 if w == w_f else 0 for w_f in word_list_formated])
train_target = [recognize(w) for w in train_data]

# Print ratio of positive words in training set
# print("Ratio of Positive data in training set:", sum([sum(x) for x in train_target]) / len(train_data))


# ======================================================================================================================
# ========================================= INPUT ENCODING & DECODING ==================================================
# Example of technique used in: https://stackoverflow.com/questions/7396849/convert-binary-to-ascii-and-vice-versa

# Here is a little demonstration that we need 5 binaries to distinguish the 26 letters [ UNCOMMENT print statements ]
alpha = 'abcdefghijklmnopqrstuvwxyz'
binaries = [bin(ord(letter_min))[-5:] for letter_min in list(alpha)]


# print(binaries)
# print(len(binaries), len(list(set(binaries))))

def encode_window(word=word_list_formated[0]):
    word_binaries = [bin(ord(l))[-5:] for l in list(word)]
    word_binaries = ''.join(word_binaries)
    word_binaries = [int(b) for b in list(word_binaries)]
    return np.array(word_binaries)


def decode_window(input_binaries_list):
    word_binaries = ''.join([str(b) for b in input_binaries_list])
    word_binaries = ['011' + word_binaries[i:i + 5] for i in range(0, len(word_binaries), 5)]
    return ''.join([chr(int(x, 2)) for x in word_binaries])


# =====================================================================================================================
# ============================================ NEURAL NETWORK =========================================================

network = MLPClassifier(hidden_layer_sizes=(10, 10), activation='relu', solver='adam', alpha=0.0001,
                        batch_size=300, learning_rate='constant', learning_rate_init=0.001, max_iter=200,
                        shuffle=True, random_state=42, tol=1e-5, verbose=True, early_stopping=False)

# Prepare train_data as 0/1 input data for model
train_data_encoded = np.zeros(window_length * 5)
train_data_encoded = [encode_window(word=train_data[i]) for i in range(len(train_data))]

# model = network.fit(train_data_encoded, train_target)

# Save model
# with lzma.open("saved_network.model", "wb") as model_file:
#    pickle.dump(model, model_file)

# Load model
with    lzma.open("saved_network.model", "rb") as model_file:
    model = pickle.load(model_file)
    with open('outputs.txt', 'w', encoding='utf-8') as f:
        i = 0
        for line in model.loss_curve_:
            f.write(f"Epoch: {i}, Global Error: {line} \n")
            i += 1
        f.write("\n")

# ======================================================================================================================
# ============================================= PREDICTIONS ============================================================

# Output decoder for decoding output of neural network and see if given input word is positive word or not
decode_output = lambda o: 'Not recognize' if np.sum(o) >= 2 or np.sum(o) == 0 else word_list[o.tolist().index(1)]

# Create Negative + Positive Words Test Set
test_size = 5000
n_p_test = [shuffle_letters(random.choice(word_list), random.randint(1, 5) * random.randint(0, 10)) \
            for _ in range(0, test_size)]

# Calculate Accuracy on Negative + Positive Words
n_p = 0
for w in n_p_test:
    pred = decode_output(model.predict(encode_window(reformat_to_window(w)).reshape(1, -1))[0])
    if w not in word_list:
        # print(w, " ======> ", pred)
        if pred != "Not recognize":
            n_p += 1
    else:
        if pred != w:
            n_p += 1

# Calculate Accuracy on Original 7 words Positive Words set
o = 0
for w in word_list:
    pred = decode_output(model.predict(encode_window(reformat_to_window(w)).reshape(1, -1))[0])
    # print(w, " ====== ", pred)
    if pred == w:
        o += 1

with open("outputs.txt", 'a', encoding='utf-8') as f:
    f.write(
        'Input                 Output                         Response                   Error      Accuracy      '
        'Reliability')
    f.write("\n")
    for i in range(len(n_p_test)):
        if n_p_test[i] in word_list:
            one = word_list.index(n_p_test[i])
            output = np.zeros(shape=(7,))
            output[one] = 1
        else:
            output = np.zeros(shape=(7,))
        pred = list(model.predict_proba(encode_window(reformat_to_window(n_p_test[i])).reshape(1, -1))[0])
        for j in range(len(pred)):
            pred[j] = float("{:.2f}".format(pred[j]))
        acc = sum(output == pred) * 100 / len(output)
        rely = sum(abs(output - pred) < 0.3) * 100 / len(output)
        f.write(
            f"{reformat_to_window(n_p_test[i])}  {output}   {pred}   {mean_squared_error(output, pred):.8f}     {acc:.2f}        {rely:.2f}")
        f.write("\n")

    f.write(f"Original Positive data accuracy: {o}/7 ({o / 7 * 100}%) \n"
      f"Negative + Positive data accuracy: {test_size - n_p}/{test_size} ({(test_size - n_p) / test_size * 100}%)")
