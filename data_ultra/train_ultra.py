from __future__ import print_function
from optparse import OptionParser
import cv2, sys, os, shutil, random
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import flip_axis, random_channel_shift
from keras.engine.training import slice_X
from keras_plus import LearningRateDecay
from u_model import get_unet, IMG_COLS as img_cols, IMG_ROWS as img_rows
from data_ultra import load_train_data, load_test_data, load_patient_num
from augmentation import random_zoom, elastic_transform, random_rotation
from utils import save_pickle, load_pickle, count_enum

_dir = os.path.join(os.path.realpath(os.path.dirname(__file__)), '')


def preprocess(imgs, to_rows=None, to_cols=None):
    if to_rows is None or to_cols is None:
        to_rows = img_rows
        to_cols = img_cols
    imgs_p = np.ndarray((imgs.shape[0], imgs.shape[1], to_rows, to_cols), dtype=np.uint8)
    for i in xrange(imgs.shape[0]):
        imgs_p[i, 0] = cv2.resize(imgs[i, 0], (to_cols, to_rows), interpolation=cv2.INTER_CUBIC)
    return imgs_p

class Learner(object):

    suffix = ''
    res_dir = os.path.join(_dir, 'res_p' + suffix)
    best_weight_path = os.path.join(res_dir, 'unet_p.hdf5')
    test_mask_res = os.path.join(res_dir, 'imgs_mask_test_p.npy')
    test_mask_exist_res = os.path.join(res_dir, 'meanstd.dump')
    validate_data_path = os.path.join(res_dir, 'valid.npy')
    tensorboard_dir = os.path.join(res_dir, 'tb')
    img_sample = os.path.join(_dir, 'IMGsample')

    def __init__(self, model_func, validation_split):
        
        self.model_func = model_func
        self.validation_split = validation_split
        self.__iter_res_dir = os.path.join(self.res_dir, 'res_iter')
        self.__iter_res_file = os.path.join(self.__iter_res_dir, '{epoch:02d}-{val_loss:.4f}.unet.hdf5')

    def _dir_init(self):
        #Initialize the result folder, if doesn't exist, create res_dir folder
        if not os.path.exists(self.res_dir):
            os.mkdir(self.res_dir)
        #check existence of iter weight files folder
        if os.path.exists(self.__iter_res_dir):
            shutil.rmtree(self.__iter_res_dir)
        os.mkdir(self.__iter_res_dir)
    
    @classmethod
    def norm_mask(cls, mask_array):
        mask_array = mask_array.astype('float32')
        mask_array /= 255.0
        return mask_array

    @classmethod
    def split_train_and_vavlid_by_patient(cls, data, mask, validation_split, shuffle=False):
        print('executed split_train_and_valid_by_patient')

        print('Shuffle & split...')
        patient_nums = load_patient_num()
        print('patient_nums: ', patient_nums)
        patient_dict = count_enum(patient_nums)
        print('patient_dict: ', patient_dict)

        x_train, y_train = (1,1)
        x_valid, y_valid = (2,2)
        return (x_train, y_train), (x_valid, y_valid)

    @classmethod
    def shuffle_train(cls, data, mask):
        perm = np.random.permutation(len(data))
        data = data[perm]
        #print ('shuffle_train data: ', data)
        mask = mask[perm]
        #print ('shuffle_train mask: ', mask)
        return data, mask

    @classmethod
    def split_train_and_valid(cls, data, mask, validation_split, shuffle=False):
        print('executed split_train_and_valid')
        print('Shuffle & split...')
        if shuffle:
            data, mask = cls.shuffle_train(data, mask)
        split_at = int(len(data) * (1. - validation_split))
        #length data: 120
        # 120 * 0.8 = 96
        #split at 96
       
        x_train, x_valid = (slice_X(data, 0, split_at), slice_X(data, mask))
        print ('data shape: ', data.shape)
        print ('x_train shape: ', x_train.shape)
        print ('y_train shape: ', x_valid.shape)

        x_valid, y_valid = (4,4)
        return (x_train, y_train), (x_valid, y_valid)

    def train_and_predict(self, pretrained_path=None, split_random=True):
        #check folder if not exist create folder
        self._dir_init() 

        print('Loading and preprocessing and standarize train data...')
        imgs_train, imgs_mask_train = load_train_data()
        #imgs_train size: (120, 1, 420, 580)
        #imgs_mask_train size: (120, 1, 420, 580)

        #j = 5
        #cv2.imwrite(os.path.join(self.img_sample, 'imgs_train_1.jpg'), imgs_train[j][0])
        #cv2.imwrite(os.path.join(self.img_sample, 'imgs_mask_train_1.jpg'), imgs_mask_train[j][0])

        imgs_train = preprocess(imgs_train)
        #cv2.imwrite(os.path.join(self.img_sample, 'imgs_preprocess_train_1.jpg'), imgs_train[j][0])
        #imgs_train preprocess size: (120, 1, 80, 112)
        
        imgs_mask_train = preprocess(imgs_mask_train)
        #cv2.imwrite(os.path.join(self.img_sample, 'imgs_process_mask_train_1.jpg'), imgs_mask_train[j][0])

        imgs_mask_train = self.norm_mask(imgs_mask_train)
        #imgs_mask_train norm_mask size: (120, 1, 420, 580)
        #cv2.imwrite(os.path.join(self.img_sample, 'imgs_mask_norm_train_1.jpg'), imgs_mask_train[j][0])
       
        split_func = split_random and self.split_train_and_valid or self.split_train_and_vavlid_by_patient 
        (x_train, y_train), (x_valid, y_valid) = split_func(imgs_train, imgs_mask_train, 
                                                            validation_split=self.validation_split)
        

def main():
    parser = OptionParser()
    parser.add_option("-s", "--split_random", action='store', type='int', dest='split_random', default = 1)
    parser.add_option("-m", "--model_name", action='store', type='str', dest='model_name', default = 'u_model')
    #
    options, _ = parser.parse_args()
    split_random = options.split_random
    model_name = options.model_name
    if model_name is None:
        raise ValueError, 'model_name is not defined'
    #
    import imp
    model_ = imp.load_source('model_', model_name + '.py')
    model_func = model_.get_unet
    #
    lr = Learner(model_func, validation_split=0.2)
    lr.train_and_predict(pretrained_path=None, split_random=split_random)
    print ('Results in ', lr.res_dir)

if __name__ == '__main__':
    sys.exit(main())