import lzma
import pickle
import random
from PIL import Image
import numpy as np
from sklearn.neural_network import MLPClassifier


class MyData:
    def __init__(self, pattern_dir):
        self.images = []
        self.symbols = [['\\', 'backslash'], ['^', 'caret'], ['-', 'dash'], ['/', 'forwardslash'],
                        ['o', 'o'], ['|', 'straightline'], ['_', 'underline']]

        # store all images into class
        for _, name in self.symbols:
            self.images.append([])
            for i in range(1, 7):
                im = Image.open(f'{pattern_dir}/{name}_{i}.png')
                im_bin = self.img_to_binary(im)
                self.images[-1].append(im_bin)

        # store image dimensionss
        im = Image.open(f'{pattern_dir}/{self.symbols[0][1]}_1.png')
        self.width, self.height = im.size

    @staticmethod
    def img_to_binary(im):
        """ converts image into binary form """
        im = im.getdata()
        im_bin = [1 if pixel == (0, 0, 0) else 0 for pixel in im]
        return im_bin

    def get_combination(self):
        """get combination for each pair of symbol-image"""
        combs = []
        for i in range(len(self.symbols)):
            for j in range(len(self.images[i])):
                combs.append([i, j])
        return combs


if __name__ == '__main__':
    # load symbols
    data = MyData('symbols')
    combinations = data.get_combination()

    inputs = []
    outputs = []

    for i in combinations:
        m, n = i
        comb = data.images[m][n]
        output = [0] * len(data.symbols)
        output[m] = 1

        # append original ones
        inputs.append(comb)
        outputs.append(output)

        # TODO: start from here
        # 25 rasteri combinations for each image
        for rasterize_combinations in range(25):
            # randomly choose shifting size for coordinates X and Y
            delta_x, delta_y = random.randint(-2, 3), random.randint(-2, 3)
            comb = [comb[data.width * i:data.width * (i + 1)] for i in range(data.height)]

            # shift pixels by X and Y coordinates
            dx, dy = [0] * abs(delta_x), [[0] * data.width] * abs(delta_y)
            comb = [r[delta_x:] + dx if delta_x >= 0 else dx + r[:delta_x] for r in comb]

            # if delta_y >= 0:
            #    comb = comb[delta_y:] + ay
            # else:
            #    ay + comb[:delta_y]
            comb = comb[delta_y:] + dy if delta_y >= 0 else dy + comb[:delta_y]

            # append both combination and respective output
            comb = sum(comb, [])
            inputs.append(comb)
            outputs.append(output)

    # convert into numpy array for usability into NN
    inputs = np.array(inputs)
    outputs = np.array(outputs)

    # ==================================================================================================================
    # ========================================== Neural Network ========================================================

    '''# initiate the network
    network = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.001,
                            batch_size=100, learning_rate='adaptive', learning_rate_init=0.001, max_iter=400,
                            shuffle=True, random_state=42, tol=1e-5, verbose=True, early_stopping=False)

    # fit the data
    model = network.fit(inputs, outputs)

    # save the network
    with lzma.open("current.model", "wb") as model_file:
        pickle.dump(model, model_file)

    # print network errors into .txt
    with open('outputs.txt', 'w', encoding='utf-8') as f:
        i = 0
        for line in network.loss_curve_:
            f.write(f"Epoch: {i}, Global Error: {line} \n")
            i += 1'''

    # OR load the saved network
    with lzma.open("saved_network.model", "rb") as model_file:
        network = pickle.load(model_file)
        with open('outputs.txt', 'w', encoding='utf-8') as f:
            i = 0
            for line in network.loss_curve_:
                f.write(f"Epoch: {i}, Global Error: {line} \n")
                i += 1

    # ==================================================================================================================
    # ============================================ Test Model ==========================================================

    # test model on images
    images = ['smile', 'smile_real', 'triangle']

    for i in images:
        im = Image.open(f'test_images/{i}.png')
        im_bin = MyData.img_to_binary(im)

        rows = []
        frame_height = int(im.height / 16)
        frame_width = int(im.width / 8)

        # get frames from test image
        for m in range(frame_height):
            rows.append([])
            for n in range(frame_width):
                p = [im_bin[r * im.width + c] for r in range(m * 16, (m + 1) * 16) for c in
                     range(n * 8, (n + 1) * 8)]
                rows[m].append(p)

        # predict output for test image
        outputs = []
        for r in rows:
            outputs.append([])
            for input in r:
                out = list(network.predict(np.array(input).reshape(1, -1))[0])
                if all(v == 0 for v in out):
                    outputs[-1].append('`')
                else:
                    outputs[-1].append(data.symbols[out.index(1)][0])

        # print output into .txt
        with open('outputs.txt', 'a', encoding='utf-8') as f:
            f.write(f"\n \t Printing --> {i} \n")
            for r in outputs:
                f.write(''.join(r) + "\n")
