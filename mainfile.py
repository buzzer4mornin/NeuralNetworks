#Generating the training data
#Taking a random list of 20 words with this website : https://www.randomlists.com/random-words?dup=false&qty=20

word_list=['nice', 'toothpaste', 'halting', 'deafening', 'structure', 'soak', 'omniscient', 'found', 'authority',
        'crabby', 'blush', 'dreary', 'childlike', 'nippy', 'roll', 'highfalutin', 'wing', 'hole', 'nonchalant', 'plant']


window_length=max([len(w) for w in word_list])+1
reformat_to_window = lambda w : w+' '*(window_length-len(w))
word_list_formated=[reformat_to_window(w) for w in word_list]


