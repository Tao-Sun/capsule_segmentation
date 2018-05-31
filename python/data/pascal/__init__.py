import numpy as np

LABELS = np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                     [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                     [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                     [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                     [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]])


HEIGHT, WIDTH = 500, 375