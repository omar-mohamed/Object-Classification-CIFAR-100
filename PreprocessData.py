import pickle
import numpy as np
import matplotlib.pyplot as plt
import random


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


train_dec = unpickle('./cifar-100-python/train')
test_dec = unpickle('./cifar-100-python/test')
meta = unpickle('./cifar-100-python/meta')


def rgb2gray(dataset):
    return np.dot(np.array(dataset, dtype='float32'), [0.299, 0.587, 0.114])


def normalization(dataset):
    # img = rgb2gray(img)
    pixel_depth = 255.0
    # mean=np.mean(dataset,axis=3)
    # std = np.std(dataset, axis=3)
    # norm_image = cv2.normalize(dataset, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return (np.array(dataset, dtype='float32') - (pixel_depth / 2)) / (pixel_depth / 2)


def reshape(dataset):
    return np.reshape(dataset, (-1, 3, 32, 32)).transpose((0, 2, 3, 1))


def randomize(dataset, labels):
    permutation = np.random.permutation(dataset.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels


def disp_sample_dataset(dataset, label, label_names, cmap=None):
    items = random.sample(range(dataset.shape[0]), 8)
    for i, item in enumerate(items):
        plt.subplot(2, 4, i + 1)
        plt.axis('off')
        plt.title(label_names[label[i]])
        plt.imshow(dataset[i, :, :], cmap=cmap)
    plt.savefig('./output_images/plt.png')
    plt.show()


train_data = train_dec[b'data']
train_labels = np.array(train_dec[b'fine_labels'])
test_data = test_dec[b'data']
test_labels = np.array(test_dec[b'fine_labels'])
label_names = meta[b'fine_label_names']

train_data = reshape(train_data)
test_data = reshape(test_data)
train_data, train_labels = randomize(train_data, train_labels)
test_data, test_labels = randomize(test_data, test_labels)
disp_sample_dataset(train_data, train_labels, label_names)

# disp_sample_dataset(rgb2gray(train_data),train_labels,label_names,'gray')

train_data = normalization(train_data)
test_data = normalization(test_data)
disp_sample_dataset(train_data, train_labels, label_names)


def get_valid_set(dataset, labels, count_per_class=20, num_classes=100):
    counter = np.zeros((num_classes), dtype='int32')
    valid_size = count_per_class * num_classes
    valid_dataset = np.zeros((valid_size, 32, 32,3), dtype='float32')
    valid_labels = np.zeros((valid_size), dtype='int32')

    valid_index = 0
    indices_to_del = []
    for i in range(dataset.shape[0]):
        if labels[i] >= count_per_class:
            continue
        valid_dataset[valid_index] = dataset[i]
        valid_labels[valid_index] = labels[i]
        indices_to_del.append(i)
        valid_index = valid_index + 1
        counter[labels[i]] = counter[labels[i]] + 1
        if (valid_index == valid_size):
            break

    dataset = np.delete(dataset, indices_to_del, axis=0)
    labels = np.delete(labels, indices_to_del)
    return dataset, labels, valid_dataset, valid_labels


train_data, train_labels, valid_data, valid_labels = get_valid_set(train_data, train_labels)

pickle_file = 'CIFAR_100_normalized.pickle'

try:
    f = open(pickle_file, 'wb')
    save = {
        'train_dataset': train_data,
        'train_labels': train_labels,
        'valid_dataset': valid_data,
        'valid_labels': valid_labels,
        'test_dataset': test_data,
        'test_labels': test_labels,
        'label_names': label_names
    }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
    print("Done")
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise
