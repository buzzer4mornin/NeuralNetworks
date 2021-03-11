import PIL.Image as IMG
import PIL.ImageOps as IMGops
import numpy as np
import os
import sys
import torch.utils
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--subtask", default=1, type=int, help="which subtask to run - 1|2|3")
parser.add_argument("--train", default=False, help="whether to train or run existing models")
parser.add_argument("--model_name", default="1st.pth", type=str, help="which saved model of specified subtask to run")
parser.add_argument("--multip", default=False, help="whether to run multiple instances")


def bin_to_im(output):
    cmap = {1: (255, 255, 255),
            0: (0, 0, 0)}
    data = [cmap[letter] for letter in output]
    im = IMG.new('RGB', (10, 10))
    im.putdata(data)
    return im


def convert_to_binary(im):
    return (np.asarray(im) * 1).flatten()


def np_to_tensor(x, y, batch_size, rotated):
    tensor = TensorDataset(torch.from_numpy(x).type(torch.FloatTensor),
                           torch.from_numpy(y).type(torch.FloatTensor))
    if not rotated:
        tensor = torch.utils.data.DataLoader(dataset=tensor, batch_size=batch_size, shuffle=False)
    return tensor


def im_to_blackwhite(im):
    thresh = 200
    fn = lambda x: 255 if x > thresh else 0
    im = im.convert('L').point(fn, mode='1')
    return im


def print_unique_outputs(unique_outputs):
    unique_outputs = dict(sorted(unique_outputs.items(), key=lambda item: item[1], reverse=True))
    c = 1
    for value in unique_outputs.values():
        print(f"Tile {c}: {value}x")
        c += 1
    '''for key, value in unique_outputs.items():
        key = key.replace("\n", "").replace("[", "").replace("]", "")
        im = bin_to_im(np.fromstring(key, dtype=int, sep=' '))
        im_resized = im.resize((100, 100))
        x_offset = 0
        comb = IMG.new('RGB', (100 * 2, 100), color="white")
        comb.paste(im_resized, (x_offset, 0))
        x_offset += im_resized.size[0]
        im2 = self.numbers[value]
        comb.paste(im2, (x_offset, 0))
        comb.show()'''


class DataSource:
    def __init__(self, tile_w, tile_h):
        self._im = self.get_image()
        self._im_w = self._im.size[0]
        self._im_h = self._im.size[1]
        self.prefab_tiles = self.get_prefab_tiles()

        self._tile_w = tile_w
        self._tile_h = tile_h

        self._input_tiles = self.get_tiles()
        self._input_b_tiles = self.get_b_tiles()
        self._input_b_tiles_rotated = self.get_b_tiles_rotated()

    @staticmethod
    def get_image():
        im = IMG.open('santa.png')
        im = im_to_blackwhite(im)
        return im

    @staticmethod
    def get_prefab_tiles():
        im_l = list()
        os.chdir(os.getcwd() + "/prefab")
        for im in os.listdir():
            if str(im).startswith(".DS_Store"):
                continue
            im = IMG.open(im)
            im = im_to_blackwhite(im)
            im_l.append(im)
        os.chdir("..")
        return im_l

    def get_tiles(self):
        l = list()
        for i in range(0, self._im_h, self._tile_h):
            for j in range(0, self._im_w, self._tile_w):
                box = (j, i, j + self._tile_w, i + self._tile_h)
                a = self._im.crop(box)
                l.append(a)
        return l

    def get_b_tiles(self):
        """get image tiles in binary format"""
        l_b = list()
        for i in range(len(self._input_tiles)):
            l_b.append(convert_to_binary(self._input_tiles[i]))
        return l_b

    def get_b_tiles_rotated(self):
        """get all 4 rotations of image tiles in binary format"""
        l_b = list()
        angle = 90
        for i in range(len(self._input_tiles)):
            im = self._input_tiles[i]
            l_b.append(convert_to_binary(im))
            for _ in range(3):
                im = im.rotate(angle)
                l_b.append(convert_to_binary(im))
        return l_b

    @property
    def input_b_tiles(self):
        return self._input_b_tiles

    @property
    def input_b_tiles_rotated(self):
        return self._input_b_tiles_rotated

    def use_prefab_or_not(self, z):
        """check whether we need to consider prefabricated tiles or not"""
        bin_im = convert_to_binary(im_to_blackwhite(z))

        # convert into half-down black
        if np.count_nonzero(bin_im[-50:] == 0) > 30 and sum(bin_im[:50]) > 30:
            # pass
            # print("yesss")
            z = self.prefab_tiles[4]
            #z = IMGops.crop(z, border=1)
            #z = IMGops.expand(z, border=1, fill=0)  # CD5C5C
            # print("half-down")
        # convert into half-up black
        if np.count_nonzero(bin_im[:50] == 0) > 30 and sum(bin_im[-50:]) > 30:
            # pass
            # print("yesss")
            z = self.prefab_tiles[5]
            #z = IMGops.crop(z, border=1)
            #z = IMGops.expand(z, border=1, fill=0)  # CD5C5C
            # print("half-up")
        # convert into white
        if sum(bin_im) > 90:
            # pass
            # print("yesss")
            z = self.prefab_tiles[7]
            # z = IMGops.crop(z, border=1)
            # z = IMGops.expand(z, border=1, fill=0)  # "#CD5C5C"
            # print("white")
        # convert into black
        if np.count_nonzero(bin_im == 0) > 85:
            # pass
            # print("yesss")
            z = self.prefab_tiles[3]
            #z = IMGops.crop(z, border=1)
            #z = IMGops.expand(z, border=1, fill=0)  # "#CD5C5C"
            # print("black")
        # convert into half-left black
        if np.count_nonzero(np.hsplit(bin_im.reshape(10, 10), 2)[0].reshape(1, -1)[0] == 0) > 25 \
                and sum(np.hsplit(bin_im.reshape(10, 10), 2)[1].reshape(1, -1)[0]) > 25:
            # pass
            z = self.prefab_tiles[0]
            # print("yess")
            # z = IMGops.crop(z, border=1)
            # z = IMGops.expand(z, border=1, fill=0)  # "#CD5C5C"
            # print("black")
        # convert into half-right black
        if sum(np.hsplit(bin_im.reshape(10, 10), 2)[0].reshape(1, -1)[0]) > 25 \
                and np.count_nonzero(np.hsplit(bin_im.reshape(10, 10), 2)[1].reshape(1, -1)[0] == 0) > 25:
            # pass
            z = self.prefab_tiles[6]
            #print("yes")
            #z = IMGops.crop(z, border=1)
            #z = IMGops.expand(z, border=1, fill=0)  # "#CD5C5C"
            # print("black")
        return z

    def get_reconstruction(self, torch_nn, rotated, prefab):
        """reconstruct and display output image by Neural Network"""
        self._im.show()
        line_im = IMG.new('RGB', (self._im_w, self._tile_h))
        final_im = IMG.new('RGB', (self._im_w, self._im_h))
        y_offset = 0
        t = 0
        rt = 0
        for i in range(0, self._im_h, self._tile_h):
            x_offset = 0
            for j in range(0, self._im_w, self._tile_w):
                # ======================================================================================================
                if prefab:
                    if i == 20 and j == 50:
                        im = self.prefab_tiles[1]
                        #im = IMGops.crop(im, border=1)
                        #im = IMGops.expand(im, border=1, fill=0)
                        line_im.paste(im, (x_offset, 0))
                        x_offset += im.size[0]
                        t += 1
                        continue

                    if i == 20 and j == 10:
                        im = self.prefab_tiles[1]
                        #im = IMGops.crop(im, border=1)
                        #im = IMGops.expand(im, border=1, fill=150)
                        line_im.paste(im, (x_offset, 0))
                        x_offset += im.size[0]
                        t += 1
                        continue

                    if i == 30 and (j == 30 or j == 20):
                        # ex = self._input_b_tiles[t]
                        # ex = ex.reshape(10, 10)
                        # if np.sum(ex) == 11: #and np.sum(ex[3]) == 10:
                        #   print("yesssss")
                        im = self.prefab_tiles[1]
                        #im = IMGops.crop(im, border=1)
                        #im = IMGops.expand(im, border=1, fill=0)
                        line_im.paste(im, (x_offset, 0))
                        x_offset += im.size[0]
                        t += 1
                        continue
                # ======================================================================================================
                if not rotated:
                    pred = (torch.round(torch_nn(torch.from_numpy(self._input_b_tiles[t].reshape(1, -1)).type(
                        torch.FloatTensor))).detach().numpy())[0]
                    im = bin_to_im(pred)
                else:
                    angle = 90
                    # take tile, pass to network, get prediction
                    original = convert_to_binary(self._input_tiles[t])
                    pred = torch_nn(torch.from_numpy(original.reshape(1, -1)).type(
                        torch.FloatTensor))

                    # get all 4 rotations of input tile and compare errors against network prediction
                    # select the one with lowest error
                    rot_0 = self._input_tiles[t]
                    rot_90 = rot_0.rotate(angle)
                    rot_180 = rot_90.rotate(angle)
                    rot_270 = rot_180.rotate(angle)

                    rot_0 = torch.from_numpy((convert_to_binary(rot_0)).reshape(1, -1)).type(
                        torch.FloatTensor)
                    rot_90 = torch.from_numpy((convert_to_binary(rot_90)).reshape(1, -1)).type(
                        torch.FloatTensor)
                    rot_180 = torch.from_numpy((convert_to_binary(rot_180)).reshape(1, -1)).type(
                        torch.FloatTensor)
                    rot_270 = torch.from_numpy((convert_to_binary(rot_270)).reshape(1, -1)).type(
                        torch.FloatTensor)

                    err = [(torch_nn.criterion(pred, rot_0)).item(), (torch_nn.criterion(pred, rot_90)).item(),
                           (torch_nn.criterion(pred, rot_180)).item(), (torch_nn.criterion(pred, rot_270)).item()]

                    # incorporate result
                    pred_binary = (torch.round(pred).detach().numpy())[0]
                    pred_im = bin_to_im(pred_binary)

                    # TODO: here
                    if sum(original) == 100 or sum(original) == 0:
                        lowest_err = 0
                    else:
                        lowest_err = np.argmin(np.array(err))

                    if lowest_err != 0:
                        rt += 1
                    if lowest_err == 0:
                        im = pred_im
                    elif lowest_err == 1:
                        im = pred_im.rotate(270)
                        #im = IMGops.crop(im, border=1)
                        #im = IMGops.expand(im, border=1, fill='red')

                    elif lowest_err == 2:
                        im = pred_im.rotate(180)
                        #im = IMGops.crop(im, border=1)
                        #im = IMGops.expand(im, border=1, fill='red')

                    else:
                        im = pred_im.rotate(90)
                        #im = IMGops.crop(im, border=1)
                        #im = IMGops.expand(im, border=1, fill='red')

                    if prefab:
                        im = self.use_prefab_or_not(im)

                line_im.paste(im, (x_offset, 0))
                x_offset += im.size[0]
                t += 1
            final_im.paste(line_im, (0, y_offset))
            y_offset += line_im.size[1]
        if rotated:
            print("Number of Rotated Tiles:", rt)
        final_im.show()
        return final_im

    def get_unique_outputs(self, torch_nn):
        """get # of unique tiles by output of Neural Network"""
        d = {}
        for i in range(len(self._input_b_tiles)):
            pred = (torch.round(torch_nn(
                torch.from_numpy(self._input_b_tiles[i].reshape(1, -1)).type(torch.FloatTensor))).detach().numpy())[0]
            try:
                d[str(pred)] += 1
            except KeyError:
                d[str(pred)] = 1
        return d


class Net(nn.Module):
    def __init__(self, in_l, h1_l, h2_l, out_l, epoch, optimizer, lr, criterion, pretrain_ratio, rotated):
        super().__init__()
        self.epoch = epoch
        self.optimizer = optimizer
        self.lr = lr
        self.criterion = criterion
        self.rotated = rotated
        self.pretrain_ratio = pretrain_ratio
        self.fc1 = nn.Linear(in_l, h1_l)
        self.fc2 = nn.Linear(h1_l, h2_l)
        self.fc3 = nn.Linear(h2_l, out_l)
        if optimizer == "Adam":
            self.optimizer = optim.Adam(self.parameters(), lr=lr)
        else:
            raise ValueError("if not Adam, implement it")

        if criterion == "MSE":
            self.criterion = nn.MSELoss()
        else:
            raise ValueError("if not MSE, implement it")

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

    def _lowest_loss(self, y_pred, y_act):
        for pred in enumerate(y_pred):
            for act in enumerate(y_act):
                if [pred[0], act[0]] == [0, 0]:
                    lowest_loss = self.criterion(pred[1], act[1])
                else:
                    curr_loss = self.criterion(pred[1], act[1])
                    if curr_loss.item() < lowest_loss.item():
                        lowest_loss = curr_loss
        return lowest_loss

    def _train(self, data):
        self.train()
        if not self.rotated:
            for t in range(self.epoch):
                err = 0
                for row in data:
                    X, y = row
                    network.zero_grad()
                    y_pred = network(X)
                    loss = self.criterion(y_pred, y)
                    loss.backward()
                    self.optimizer.step()
                    err += loss.item()
                print(f"Epoch {t}  Global error  {(err / len(data)):.5f}")
        else:
            for t in range(int(self.epoch * self.pretrain_ratio)):
                err = 0
                for row in data:
                    X, y = row
                    network.zero_grad()
                    y_pred = network(X)
                    loss = self.criterion(y_pred, y)
                    loss.backward()
                    self.optimizer.step()
                    err += loss.item()
                print(f"Epoch {t}  Global error  {(err / len(data)):.5f}  -- Allow Network To Recognize All Rotations")
            for t in range(int(self.epoch * self.pretrain_ratio), self.epoch):
                err = 0

                for row in range(0, len(data), 4):
                    network.zero_grad()

                    x0, y0 = data[row]
                    x1, y1 = data[row + 1]
                    x2, y2 = data[row + 2]
                    x3, y3 = data[row + 3]

                    y_pred = [network(x0), network(x1), network(x2), network(x3)]
                    y_act = [y0, y1, y2, y3]

                    lowest_loss = self._lowest_loss(y_pred, y_act)
                    lowest_loss.backward()
                    self.optimizer.step()
                    err += lowest_loss.item()
                print(f"Epoch {t}  Global error  {(err / len(data)):.5f}")


if __name__ == '__main__':
    # TERMINAL ARGUMENTS
    args = parser.parse_args([] if "__file__" not in globals() else None)
    rotation_bool = False if args.subtask == 1 else True
    prefab_bool = True if args.subtask == 3 else False

    # LOAD Santa image and pre-process it
    santa = DataSource(tile_w=10, tile_h=10)

    # TRAIN NEW MODEL or USE EXISTING ONE
    if args.train:
        # initialize network
        '''+{best} - epoch(200), hidden(7,4), batch(2), lr(0.001), adam                                         [1st-sub]
           {best2} - epoch(300), hidden(7,4), lr(0.003), ratio(0.1), adam                                      [2nd-sub]
           {best1} - epoch(200), hidden(7,4/3), lr(0.0007), ratio(0.2/0.15), adam (run repetitively,check results)  [2nd-sub]
           +{best0} - epoch(250), hidden(8,4), lr(0.0007), ratio(0.15), adam (run repetitively,check results)  [2nd-sub]'''

        network = Net(in_l=100, h1_l=8, h2_l=4, out_l=100, epoch=250, optimizer="Adam",
                      lr=0.0007, criterion="MSE", pretrain_ratio=0.15, rotated=rotation_bool)

        # get data and train network
        b_input = b_output = np.array(santa.input_b_tiles_rotated) if args.subtask != 1 \
            else np.array(santa._input_b_tiles)
        my_data = np_to_tensor(b_input, b_output, batch_size=2, rotated=rotation_bool)
        network._train(data=my_data)
    else:
        model_path = str(os.getcwd()) + "/models/" + str(args.subtask) + "/" + args.model_name
        network = torch.load(model_path)

    # RECONSTRUCT IMAGE
    network.eval()
    reconstructed_santa = santa.get_reconstruction(network, rotated=rotation_bool, prefab=prefab_bool)

    # PRINT RESULTS
    print(f"Number of Unique tiles: {len(santa.get_unique_outputs(network))}")
    print(f"Total Price: {len(santa.get_unique_outputs(network)) * 1000 + 1400} $$")
    print(f"Net Profit: {60000 - (len(santa.get_unique_outputs(network)) * 1000 + 1400)} $$")
    unique_outputs = santa.get_unique_outputs(network)
    if not prefab_bool:
        print_unique_outputs(unique_outputs)

    # SAVE NEWLY TRAINED MODEL
    # train multiple times:
    # for ((i=1; i<=2; i++)); do python train.py --subtask=2 --train=True --multip=True; done
    random_model = str(np.random.randint(999999))

    if args.train and args.multip:
        os.chdir(os.getcwd() + "/2nd_randoms")
        curr_dir = str(os.getcwd()) + "/" + random_model
        torch.save(network, curr_dir)
        os.chdir("..")
    else:
        save_dirr = str(os.getcwd()) + "/current.pth"
        torch.save(network, save_dirr)
