import os
import random
from SSD.utils import create_data_lists

image_directory = 'data/SmartWarehouse/JPEGImages'
trainval_file = 'data/SmartWarehouse/ImageSets/Main/trainval.txt'
test_file = 'data/SmartWarehouse/ImageSets/Main/test.txt'

def read_data_files():
    return os.listdir(image_directory)

def k_fold_cross_validation(k_fold):
    files = read_data_files()
    random.shuffle(files)
    folds = []

    for i in range(k_fold):
        folds.append([])

    for index, file in enumerate(files):
        folds[index % k_fold].append(file)

    return folds

def specify_train_test_data(train_images, test_images):
    global trainval_file
    global test_file
    trainval = open(trainval_file, "w")
    test = open(test_file, "w")
    trainval_file_text = ''
    test_file_text = ''

    for index, train_image in enumerate(train_images):
        trainval_file_text += ''.join(train_image.replace(".jpg","")) + "\n"
    for index, test_image in enumerate(test_images):
        test_file_text += ''.join(test_image.replace(".jpg","")) + "\n"

    trainval.write(trainval_file_text)
    trainval.close()

    test.write(test_file_text)
    test.close()

    create_data_lists(smartwarehouse_path='data/SmartWarehouse', output_folder='data/Output')

if __name__ == '__main__':
    for index, file in enumerate(read_data_files()):
        os.rename(os.path.join(image_directory, file), os.path.join(image_directory, ''.join([str(index + 1).zfill(6), '.jpg'])))