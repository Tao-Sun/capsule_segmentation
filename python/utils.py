import cv2
import numpy as np
from sklearn.metrics import confusion_matrix


def dice(target_img, prediction_img, smooth=1.0):
    target = target_img.flatten()
    prediction = prediction_img.flatten()
    intersection = np.sum(np.multiply(target, prediction))
    union = np.sum(target) + np.sum(prediction)
    dice = (2.0 * intersection + smooth) / (union + smooth)

    return dice

def get_confusion_matrix(target, prediction):
    labels = np.unique(target)
    confusion_mat = confusion_matrix(target.flatten(),
                                     prediction.flatten(),
                                     labels=labels)
    return confusion_mat


def dice_ratio(target, prediction):
    confusion_mat = get_confusion_matrix(target, prediction)
    true_positives = np.float64(confusion_mat.diagonal())

    class_actual_sums = np.sum(confusion_mat, axis=1)
    false_negative = class_actual_sums - true_positives

    class_predicted_sums = np.sum(confusion_mat, axis=0)
    false_positive = class_predicted_sums - true_positives

    dice_ratio = (2*true_positives) / (2*true_positives + false_negative + false_positive)
    return dice_ratio


def accuracy_stats(target_vec, prediction_vec, labels):
    val = []

    confusion = confusion_matrix(target_vec, prediction_vec, labels=labels)
    true_positives = confusion.diagonal()
    class_actual_sums = np.sum(confusion, axis=1)
    class_predicted_sums = np.sum(confusion, axis=0)
    false_positives = class_predicted_sums - true_positives

    val.append(true_positives)
    val.append(class_actual_sums)
    val.append(false_positives)
    return np.array(val)


def accuracies(true_positives, class_sums, false_positives):
    print('true_positives length:' + str(true_positives))
    true_positives = true_positives.astype(np.float32)
    class_sums = class_sums.astype(np.float32)
    false_positives = false_positives.astype(np.float32)

    global_accuracy = np.sum(true_positives) / np.sum(class_sums)

    class_accuracies = true_positives/class_sums

    mIoUs = true_positives / (class_sums + false_positives)
    # mIoUs = (2 * true_positives) / (class_sums + false_positives + true_positives)
    # mIoU = np.nanmean(mIoUs)
    return global_accuracy, class_accuracies, mIoUs


def add_noise(img, low, high):
    noise = cv2.randu(np.zeros(img.shape, np.uint8), low, high)
    img = cv2.add(img, noise)
    return img


def connected_error_num(target, prediction, target_label, threshold=35, connectivity=8):
    """
    Number of connected blocks of error labels whose area is
    larger than the threshold.

    :param target:
    :param prediction:
    :param target_label:
    :param connectivity:
    :param threshold:
    :return:
    """

    assert(target.shape == prediction.shape)
    target = target.astype(np.uint8)
    # print('target')
    # print(target)
    prediction = prediction.astype(np.uint8)
    # print('prediction')
    # print(prediction)
    target_label = int(target_label)

    target_shape = target.shape
    background_value = 0

    error_map_1 = np.zeros(target_shape, dtype=np.uint8)
    error_map_2 = np.zeros(target_shape, dtype=np.uint8)
    error_map_3 = np.zeros(target_shape, dtype=np.uint8)
    error_map_1[np.where(target == target_label)] = 1
    error_map_2[np.where(prediction != target_label)] = 1
    error_map_3[np.where(prediction != background_value)] = 1

    error_map = np.logical_and(error_map_1, error_map_2)
    error_map = np.logical_and(error_map, error_map_3).astype(np.uint8)
    # print(error_map)
    _, _, stats, _ = cv2.connectedComponentsWithStats(error_map, connectivity)
    error_areas = stats[1:, 4]
    error_num = len(np.where(error_areas > threshold)[0])
    return error_num
