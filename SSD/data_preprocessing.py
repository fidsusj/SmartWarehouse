import os
import random
from SSD.utils import create_data_lists

dataset_path = 'D:/Workspaces/PyCharmProjects/SmartWarehouse/SSD/data/SmartWarehouseSSD/'
image_directory = dataset_path + 'JPEGImages'
train_file = dataset_path + 'ImageSets/Main/train.txt'
test_file = dataset_path + 'ImageSets/Main/test.txt'
validation_file = dataset_path + 'ImageSets/Main/validation.txt'


def read_data_files():
    return os.listdir(image_directory)


def k_fold_cross_validation(k_fold):
    files = read_data_files()
    train_test = files[:998]
    validation = files[998:]
    random.shuffle(train_test)
    folds = []

    for i in range(k_fold):
        folds.append([])

    for index, file in enumerate(train_test):
        folds[index % k_fold].append(file)

    return folds, validation


def specify_train_test_validation_data(train_images, test_images, validation_images):
    global train_file
    global test_file
    global validation_file
    train = open(train_file, "w")
    test = open(test_file, "w")
    validation = open(validation_file, "w")
    train_file_text = ''
    test_file_text = ''
    validation_file_text = ''

    for _, train_image in enumerate(train_images):
        train_file_text += ''.join(train_image.replace(".jpg", "")) + "\n"
    for _, test_image in enumerate(test_images):
        test_file_text += ''.join(test_image.replace(".jpg", "")) + "\n"
    for _, validation_image in enumerate(validation_images):
        validation_file_text += ''.join(validation_image.replace(".jpg", "")) + "\n"

    train.write(train_file_text)
    train.close()

    test.write(test_file_text)
    test.close()

    validation.write(validation_file_text)
    validation.close()

    create_data_lists(smartwarehouse_path=dataset_path, output_folder='data/Output')


if __name__ == '__main__':
    for index, file in enumerate(read_data_files()):
        os.rename(os.path.join(image_directory, file),
                  os.path.join(image_directory, ''.join([str(index + 1).zfill(6), '.jpg'])))
