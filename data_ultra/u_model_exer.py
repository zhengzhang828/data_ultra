import sys
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Dense
from keras.layers import BatchNormalization, Dropout, Flatten, Lambda
from keras.layers.advanced_activations import ELU, LeakyReLU
from metric import dice_coef, dice_coef_loss

IMG_ROWS, IMG_COLS = 80, 112

#github commit change testing

def inception_block(inputs, depth, batch_mode=0, splitted=False, activation='relu'):
    assert depth % 16 == 0
    actv = activation == 'relu' and (lambda: LeakyReLU(0.0)) or activation == 'elu' and (lambda: ELU(1.0)) or None

    c1_1 = Convolution2D(depth/4, 1, 1, init='he_normal', border_mode='same')(inputs)
    
    c2_1 = Convolution2D(depth/8*3, 1, 1, init='he_normal', border_mode='same')(inputs)

    c2_1 = actv()(c2_1)

    if splitted:
        c2_2 = Convolution2D(depth/2, 1, 3, init='he_normal', border_mode='same')(c2_1)
        c2_2 = BatchNormalization(mode=batch_mode, axis=1)(c2_2)
        c2_2 = actv()(c2_2)
        c2_3 = Convolution2D(depth/2, 3, 1, init='he_normal', border_mode='same')(c2_2)
    else:
        c2_3 = Convolution2D(depth/2, 3, 3, init='he_normal', border_mode='same')(c2_1)

    c3_1 = Convolution2D(depth/16, 1, 1, init='he_normal', border_mode='same')(inputs)
    c3_1 = actv()(c3_1)

    if splitted:
        c3_2 = Convolution2D(depth/8, 1, 5, init='he_normal', border_mode='same')(c3_1)
        c3_2 = BatchNormalization(mode=batch_mode, axis=1)(c3_2)
        c3_2 = actv()(c3_2)
        c3_3 = Convolution2D(depth/8, 5, 1, init='he_normal', border_mode='same')(c3_2)
    else:
        c3_3 = Convolution2D(depth/8, 5, 5, init='he_normal', border_mode='same')(c3_1)

    p4_1 = MaxPooling2D(pool_size=(3,3), strides=(1,1), border_mode='same')(inputs)
    c4_2 = Convolution2D(depth/8, 1, 1, init='he_normal', border_mode='same')(p4_1)

    res = merge([c1_1, c2_3, c3_3, c4_2], mode='concat', concat_axis=1)
    res = BatchNormalization(mode=batch_mode, axis=1)(res)
    res = actv()(res)

    return res

def _shortcut(_input, residual):
    stride_width = _input._keras_shape[2] / residual._keras_shape[2]
    stride_height = _input._keras_shape[3] / residual._keras_shape[3]
    equal_channels = residual._keras_shape[1] == _input._keras_shape[1]

    shortcut = _input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Convolution2D(nb_filter=residual._keras_shape[1], nb_row=1, nb_col=1,
                                 subsample=(stride_width, stride_height),
                                 init="he_normal", border_mode="valid")(_input)

    return merge([shortcut, residual], mode="sum")

def rblock(inputs, num, depth, scale=0.1):    
    residual = Convolution2D(depth, num, num, border_mode='same')(inputs)
    residual = BatchNormalization(mode=2, axis=1)(residual)
    residual = Lambda(lambda x: x*scale)(residual)
    res = _shortcut(inputs, residual)
    return ELU()(res) 

def NConvolution2D(nb_filter, nb_row, nb_col, border_mode='same', subsample=(1, 1)):
    def f(_input):
        conv = Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                              border_mode=border_mode)(_input)
        norm = BatchNormalization(mode=2, axis=1)(conv)
        return ELU()(norm)

    return f

def get_unet_inception_2head_(optimizer):
    splitted = True
    act = 'elu'

    inputs = Input((1, IMG_ROWS, IMG_COLS), name='main_input')
    conv1 = inception_block(inputs, 32, batch_mode=2, splitted=splitted, activation=act)

    pool1 = NConvolution2D(32, 3, 3, border_mode='same', subsample=(2,2))(conv1)
    pool1 = Dropout(0.5)(pool1) 

    conv2 = inception_block(pool1, 64, batch_mode=2, splitted=splitted, activation=act)
    pool2 = NConvolution2D(64, 3, 3, border_mode='same', subsample=(2,2))(conv2)
    pool2 = Dropout(0.5)(pool2)

    conv3 = inception_block(pool2, 128, batch_mode=2, splitted=splitted, activation=act)
    pool3 = NConvolution2D(128, 3, 3, border_mode='same', subsample=(2,2))(conv3)
    pool3 = Dropout(0.5)(pool3)

    conv4 = inception_block(pool3, 256, batch_mode=2, splitted=splitted, activation=act)
    pool4 = NConvolution2D(128, 3, 3, border_mode='same', subsample=(2,2))(conv4)
    pool4 = Dropout(0.5)(pool4)

    conv5 = inception_block(pool4, 512, batch_mode=2, splitted=splitted, activation=act)
    conv5 = Dropout(0.5)(conv5)

    #Flatten output
    pre = Convolution2D(1, 1, 1, init='he_normal', activation='sigmoid')(conv5)
    pre = Flatten()(pre)
    aux_out = Dense(1, activation='sigmoid', name='aux_output')(pre)

    after_conv4 = rblock(conv4, 1, 256)
    up6 = merge([UpSampling2D(size=(2, 2))(conv5), after_conv4], mode='concat', concat_axis=1)
    conv6 = inception_block(up6, 256, batch_mode=2, splitted=splitted, activation=act)

    model = Model(input=inputs, output=conv6)

    return model

def get_unet_inception_2head(optimizer):
    splitted = True
    act = 'elu'
    
    inputs = Input((1, IMG_ROWS, IMG_COLS), name='main_input')
    conv1 = inception_block(inputs, 32, batch_mode=2, splitted=splitted, activation=act)
    #conv1 = inception_block(conv1, 32, batch_mode=2, splitted=splitted, activation=act)
    
    #pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = NConvolution2D(32, 3, 3, border_mode='same', subsample=(2,2))(conv1)
    pool1 = Dropout(0.5)(pool1)
    
    conv2 = inception_block(pool1, 64, batch_mode=2, splitted=splitted, activation=act)
    #pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = NConvolution2D(64, 3, 3, border_mode='same', subsample=(2,2))(conv2)
    pool2 = Dropout(0.5)(pool2)
    
    conv3 = inception_block(pool2, 128, batch_mode=2, splitted=splitted, activation=act)
    #pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = NConvolution2D(128, 3, 3, border_mode='same', subsample=(2,2))(conv3)
    pool3 = Dropout(0.5)(pool3)
     
    conv4 = inception_block(pool3, 256, batch_mode=2, splitted=splitted, activation=act)
    #pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = NConvolution2D(256, 3, 3, border_mode='same', subsample=(2,2))(conv4)
    pool4 = Dropout(0.5)(pool4)
    
    conv5 = inception_block(pool4, 512, batch_mode=2, splitted=splitted, activation=act)
    #conv5 = inception_block(conv5, 512, batch_mode=2, splitted=splitted, activation=act)
    conv5 = Dropout(0.5)(conv5)
    
    #
    pre = Convolution2D(1, 1, 1, init='he_normal', activation='sigmoid')(conv5)
    pre = Flatten()(pre)
    aux_out = Dense(1, activation='sigmoid', name='aux_output')(pre) 
    #
    
    #Upsampling matching the convolutional neural network
    after_conv4 = rblock(conv4, 1, 256)
    up6 = merge([UpSampling2D(size=(2, 2))(conv5), after_conv4], mode='concat', concat_axis=1)
    conv6 = inception_block(up6, 256, batch_mode=2, splitted=splitted, activation=act)
    conv6 = Dropout(0.5)(conv6)

    after_conv3 = rblock()

    #model = Model(input=inputs, output=conv6)

    model = Model(input=inputs, output=up6)

    return model

get_unet = get_unet_inception_2head

def main():
    
    from keras.optimizers import Adam, RMSprop, SGD
    from metric import dice_coef, dice_coef_loss

    img_rows = IMG_ROWS
    img_cols = IMG_COLS

    optimizer = RMSprop(lr=0.045, rho=0.9, epsilon=1.0)
    model = get_unet(Adam(lr=1e-5))

    model.compile(optimizer=optimizer, loss=dice_coef_loss, metrics=[dice_coef])

    from keras.utils.visualize_util import plot
    plot(model, to_file='model_inception.png')

if __name__ == '__main__':
    sys.exit(main())
    