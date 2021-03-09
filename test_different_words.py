import random
import numpy as np
import lzma
import pickle
from sklearn.metrics import mean_squared_error

word_list = ['nice', 'toothpaste', 'highfalutin', 'childlike', 'authority', 'halting', 'found']
window_length = max([len(w) for w in word_list]) + 1
reformat_to_window = lambda w: w + ' ' * (window_length - len(w))
decode_output = lambda o: 'Not recognize' if np.sum(o) >= 2 or np.sum(o) == 0 else word_list[o.tolist().index(1)]


def encode_window(word):
    word_binaries = [bin(ord(l))[-5:] for l in list(word)]
    word_binaries = ''.join(word_binaries)
    word_binaries = [int(b) for b in list(word_binaries)]
    return np.array(word_binaries)


def decode_window(input_binaries_list):
    word_binaries = ''.join([str(b) for b in input_binaries_list])
    word_binaries = ['011' + word_binaries[i:i + 5] for i in range(0, len(word_binaries), 5)]
    return ''.join([chr(int(x, 2)) for x in word_binaries])


def shuffle_letters(word, shuffle_letterse=5):
    """ Recursive function that exchange two letters at each iteration (shuffle)"""
    if type(word) is not list:
        word = list(word)
    if shuffle_letterse == 0:
        return ''.join(word)  # word==shuffle_letters(word,shuffle=0) -> TRUE
    i, j = random.randint(0, len(word) - 1), random.randint(0, len(word) - 1)
    word[i], word[j] = word[j], word[i]
    return shuffle_letters(word, shuffle_letterse - 1)


with lzma.open("saved_network.model", "rb") as model_file:
    model = pickle.load(model_file)
# ======================================================================================================================
# =========================================== TEST ON DIFFERENT WORDS ==================================================


test_size = 5000
diff_words = ["scold", "doubt", "verdant", "describe", "wretched", "lopsided", "medical", "disturbed", "welcome",
              "decision"]
diff_words_test = [shuffle_letters(random.choice(diff_words), random.randint(1, 5) * random.randint(0, 10)) \
                   for _ in range(0, test_size)]

# Calculate Accuracy on Different Words Dataset
neg = 0
wrongly_classified_words = {}
for w in diff_words_test:
    pred = decode_output(model.predict(encode_window(reformat_to_window(w)).reshape(1, -1))[0])
    if w not in word_list:
        if pred != "Not recognize":
            wrongly_classified_words[str(w)] = pred
            neg += 1
    else:
        if pred != w:
            wrongly_classified_words[str(w)] = pred
            neg += 1

print(wrongly_classified_words)

with open("outputs_different_words.txt", 'w', encoding='utf-8') as f:
    f.write(
        'Input                 Output                         Response                   Error      Accuracy      '
        'Reliability')
    f.write("\n")
    for i in range(len(diff_words_test)):
        if diff_words_test[i] in word_list:
            one = word_list.index(diff_words_test[i])
            output = np.zeros(shape=(7,))
            output[one] = 1
        else:
            output = np.zeros(shape=(7,))
        pred = list(model.predict_proba(encode_window(reformat_to_window(diff_words_test[i])).reshape(1, -1))[0])
        for j in range(len(pred)):
            pred[j] = float("{:.2f}".format(pred[j]))
        acc = sum(output == pred) * 100 / len(output)
        rely = sum(abs(output - pred) < 0.3) * 100 / len(output)
        f.write(
            f"{reformat_to_window(diff_words_test[i])}  {output}   {pred}   {mean_squared_error(output, pred):.8f}     {acc:.2f}        {rely:.2f}")
        f.write("\n")

    f.write(f"\nDifferent Words Dataset: {test_size - neg}/{test_size} ({(test_size - neg) / test_size * 100}%)\n")

    f.write(f"\n{neg} words are wrongly classified: \n")

    for i, j in wrongly_classified_words.items():
        f.write(f"-''{i}'' is wrongly classified as ''{j}''\n")