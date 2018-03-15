import numpy as np
from matplotlib import pyplot as plt
import scipy.io as sio
import sys
import random


data = sio.loadmat('validation.mat')
images = data['affNISTdata'][0][0][2]
labels = data['affNISTdata'][0][0][5]


def show(img_1, img_2, combined_1, combined_2):
    fig = plt.figure(figsize=(1, 4))

    fig.add_subplot(1, 4, 1)
    plt.imshow(img_1, cmap = 'gray', interpolation = 'bicubic')
    fig.add_subplot(1, 4, 2)
    plt.imshow(img_2, cmap = 'gray', interpolation = 'bicubic')
    fig.add_subplot(1, 4, 3)
    plt.imshow(combined_1, cmap='gray', interpolation='bicubic')
    fig.add_subplot(1, 4, 4)
    plt.imshow(combined_2, cmap='gray', interpolation='bicubic')

    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()


def get_zone(img):
    row_range = (np.nonzero(img)[0].min(), np.nonzero(img)[0].max())
    col_range = (np.nonzero(img)[1].min(), np.nonzero(img)[1].max())

    zone = (row_range, col_range)
    return zone


def combine(framed_1, framed_2):
    h1, w1 = framed_1.shape
    h2, w2 = framed_2.shape

    def get_combine_dim():
        if (h1 + h2) <= 40 and (w1 + w2) > 40:
            return 0
        elif (h1 + h2) > 40 and (w1 + w2) <= 40:
            return 1
        elif (h1 + h2) <= 40 and (w1 + w2) <= 40:
            return random.randint(0, 1)
        elif (h1 + h2) > 40 and (w1 + w2) > 40:
            return None

    def get_padding(total_padding):
        before = int(random.uniform(0, 1) * total_padding)
        after = total_padding - before
        return before, after

    combine_dim = get_combine_dim()
    if combine_dim is not None:
        if combine_dim == 0:
            if h1 <= 20 and h2 <=20:
                total_padding_d1 = 40 - h1 - h2
                total_padding1_d1 = int(random.uniform(0, 1) * (total_padding_d1))
                total_padding2_d1 = total_padding_d1 - total_padding1_d1
            elif h1 >20:
                total_padding1_d1 = 0
                total_padding2_d1 = 40 - h1 - h2
            elif h2 > 20:
                total_padding1_d1 = 40 - h1 - h2
                total_padding2_d1 = 0

            padding1_d1 = get_padding(total_padding1_d1)
            total_padding1_d2 = 40 - w1
            padding1_d2 = get_padding(total_padding1_d2)
            padded_1 = np.pad(framed_1,
                              (padding1_d1, padding1_d2),
                              'constant', constant_values=((0, 0), (0, 0)))


            padding2_d1 = get_padding(total_padding2_d1)
            total_padding2_d2 = 40 - w2
            padding2_d2 = get_padding(total_padding2_d2)
            padded_2 = np.pad(framed_2,
                              (padding2_d1, padding2_d2),
                              'constant', constant_values=((0, 0), (0, 0)))

            return np.vstack((padded_1, padded_2))

        elif combine_dim == 1:
            if w1 <= 20 and w2 <=20:
                total_padding_d2 = 40 - w1 - w2
                total_padding1_d2 = int(random.uniform(0, 1) * (total_padding_d2))
                total_padding2_d2 = total_padding_d2 - total_padding1_d2
            elif w1 >20:
                total_padding1_d2 = 0
                total_padding2_d2 = 40 - w1 - w2
            elif w2 > 20:
                total_padding1_d2 = 40 - w1 - w2
                total_padding2_d2 = 0


            total_padding1_d1 = 40 - h1
            padding1_d1 = get_padding(total_padding1_d1)

            padding1_d2 = get_padding(total_padding1_d2)
            padded_1 = np.pad(framed_1,
                              (padding1_d1, padding1_d2),
                              'constant', constant_values=((0, 0), (0, 0)))

            total_padding2_d1 = 40 - h2
            padding2_d1 = get_padding(total_padding2_d1)
            padding2_d2 = get_padding(total_padding2_d2)
            padded_2 = np.pad(framed_2,
                              (padding2_d1, padding2_d2),
                              'constant', constant_values=((0, 0), (0, 0)))

            return np.hstack((padded_1, padded_2))
    else:
        return None


num = 0

num_0 = []
num_1 = []
num_0_orig = []
num_1_orig = []
for i in range(0, 32):
    print('number: %d' % labels[:, i])
    img = np.reshape(images[:, i], (40, 40))
    zone = get_zone(img)
    framed = img[zone[0][0]:zone[0][1], zone[1][0]:zone[1][1]]
    #show(img, img[zone[0][0]:zone[0][1], zone[1][0]:zone[1][1]])
    print(framed)

    rows = zone[0][1]-zone[0][0]
    cols = zone[1][1]-zone[1][0]
    print('zone: (%d, %d)' % (rows, cols))

    if labels[:, i] == 1 or labels[:, i] == 3:
        num += 1

    if i % 2 == 0:
        num_0.append(framed)
        num_0_orig.append(img)
    else:
        num_1.append(framed)
        num_1_orig.append(img)

combined_1 = combine(num_0[len(num_0)-1], num_1[len(num_1)-1])
combined_2 = combine(num_0[len(num_0)-1], num_1[len(num_1)-1])

if combined_1 is not None:
    show(num_0[len(num_0)-1], num_1[len(num_1)-1], combined_1, combined_2)
else:
    print("combine failure!")


print('Bigger num: %d' % num)
