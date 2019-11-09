import os
from SSD.utils import create_data_lists

image_directory = 'data/SmartWarehouse/JPEGImages'
trainval_file = 'data/SmartWarehouse/ImageSets/Main/trainval.txt'
test_file = 'data/SmartWarehouse/ImageSets/Main/test.txt'

if __name__ == '__main__':
    files = os.listdir(image_directory)
    trainval_file = open(trainval_file, "w")
    test_file = open(test_file, "w")
    trainval_file_text = ''
    test_file_text = ''

    for index, file in enumerate(files):
        #os.rename(os.path.join(image_directory, file), os.path.join(image_directory, ''.join([str(index + 1).zfill(6), '.jpg'])))
        if(index % 20 >= 16):
            test_file_text += ''.join([str(index + 1).zfill(6), '\n'])
        else:
            trainval_file_text += ''.join([str(index + 1).zfill(6), '\n'])

    trainval_file.write(trainval_file_text)
    trainval_file.close()

    test_file.write(test_file_text)
    test_file.close()

    create_data_lists(smartwarehouse_path='data/SmartWarehouse',
                      output_folder='data/Output')
