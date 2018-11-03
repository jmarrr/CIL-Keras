import numpy as np
import glob
import h5py
import itertools

import argparse

import imgaug as ia
from imgaug import augmenters as iaa

from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback
from keras.optimizers import Adam
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, concatenate


def train_CIL(args):
    # Load Dataset
    trainPath = './dataset/SeqTrain/'
    valPath = './dataset/SeqVal/'

    trainFile = glob.glob(trainPath + '*.h5')
    valFile = glob.glob(valPath + '*.h5')

    # Data Augmentation
    st = lambda aug: iaa.Sometimes(0.4, aug)
    oc = lambda aug: iaa.Sometimes(0.3, aug)
    rl = lambda aug: iaa.Sometimes(0.09, aug)

    seq = iaa.Sequential([
            rl(iaa.GaussianBlur((0, 1.5))),                                               # blur images with a sigma between 0 and 1.5
            rl(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05), per_channel=0.5)),     # add gaussian noise to images
            oc(iaa.Dropout((0.0, 0.10), per_channel=0.5)),                                # randomly remove up to X% of the pixels
            oc(iaa.CoarseDropout((0.0, 0.10), size_percent=(0.08, 0.2),per_channel=0.5)), # randomly remove up to X% of the pixels
            oc(iaa.Add((-40, 40), per_channel=0.5)),                                      # adjust brightness of images (-X to Y% of original value)
            st(iaa.Multiply((0.10, 2.5), per_channel=0.2)),                               # adjust brightness of images (X -Y % of original value)
            rl(iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5)),                   # adjust the contrast
    ], random_order=True)

    # Configuration
    branch_config = [
                        ["Speed"],
                        ["Steer", "Gas", "Brake"], ["Steer", "Gas", "Brake"], 
                        ["Steer", "Gas", "Brake"], ["Steer", "Gas", "Brake"]    
                    ]

    branch_names = ['Speed', 'Follow', 'Left', 'Right', 'Straight']

    image_size = (88, 200, 3)

    input_image = (image_size[0], image_size[1], image_size[2])

    input_speed = (1,)

    batch_size = args.batch_size

    weights_path = args.weights

    # High level command: 1 - Speed, 2 - Follow lane, 3 - Go Left, 4 - Go Right, 5 - Straight, None - Train all branches
    # masks = args.masks

    # Generator
    def batch_generator(file_names, batch_size = 6, masks = None):  
    
        ''' High level command: 2 - Follow lane, 3 - Left, 4 - Right, 5 - Straight '''
        
        batch_x = []   
        batch_y = []
        batch_s = []
        
        while True:
            for i in range(batch_size - 1):
                file_idx = np.random.randint(len(file_names) - 1)
                sample_idx = np.random.randint(200-1)

                data = h5py.File(file_names[file_idx], 'r')

                for mask in masks:
                    if data['targets'][sample_idx][24] == mask:
                        batch_x.append(seq.augment_image(data['rgb'][sample_idx]))
                        batch_y.append(data['targets'][sample_idx][:3])
                        batch_s.append(data['targets'][sample_idx][10]) # speed
                        
                data.close()
                
            yield ([np.array(batch_x), np.array(batch_s)], [np.array(batch_s) if mask == 1 else np.array(batch_y) for mask in masks ])
    

    # Network
    def CIL(input_image, input_speed, masks = None, weights_path = None):
        """
           Parameters
           ----------
                input_image : tuple
                    Image input shape.
                    
                input_speed : tuple
                    Speed measurements input shape.
                    
                masks : list
                    Index of branch to be trained.
                    
                weights_path : string 
                    Path to the weights file
                    
            Returns
            -------
                Model : keras-object
                    Network object to be trained
                    
        """
        
        branches = []

        def conv_block(inputs, filters, kernel_size, strides):
            x = Conv2D(filters, (kernel_size, kernel_size), strides = strides, activation='relu')(inputs)
            x = MaxPooling2D(pool_size=(1,1), strides=(1,1))(x)
            x = BatchNormalization()(x)
            x = Dropout(0.5)(x)

            return x

        def fc_block(inputs, units):
            fc = Dense(units, activation = 'relu')(inputs)
            fc = Dropout(0.2)(fc)

            return fc

        xs = Input(input_image, name='image_input')

        '''inputs, filters, kernel_size, strides'''

        """ Conv 1 """
        x = conv_block(xs, 32, 5, 2)
        x = conv_block(x, 32, 3, 1)

        """ Conv 2 """
        x = conv_block(x, 64, 3, 2)
        x = conv_block(x, 64, 3, 1)

        """ Conv 3 """
        x = conv_block(x, 128, 3, 2)
        x = conv_block(x, 128, 3, 1)

        """ Conv 4 """
        x = conv_block(x, 256, 3, 1)
        x = conv_block(x, 256, 3, 1)

        """ Reshape """
        x = Flatten()(x)

        """ FC1 """
        x = fc_block(x, 512)

        """ FC2 """
        x = fc_block(x, 512)

        """Process Control"""

        """ Speed (measurements) """
        sm = Input(input_speed, name='speed_input') 
        speed = fc_block(sm, 128)
        speed = fc_block(speed, 128)

        """ Joint sensory """
        j = concatenate([x, speed])
        j = fc_block(j, 512)


        for i in range(len(branch_config)):

            if branch_config[i][0] == "Speed":
                branch_output = fc_block(x, 256)
                branch_output = fc_block(branch_output, 256)

            else:
                branch_output = fc_block(j, 256)
                branch_output = fc_block(branch_output, 256)

            fully_connected = Dense(len(branch_config[i]), name = branch_names[i])(branch_output)
            branches.append(fully_connected)
        
        # Load weights
        if weights_path:
            model.load_weights(weights_path)
            
        return model

    model = CIL(input_image, input_speed, masks, weights_path)

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.00003), metrics=['accuracy'])

    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                     monitor='val_loss',
                                     verbose=0,
                                     save_best_only=True,
                                     mode='auto')

    model.fit_generator(batch_generator(trainFile, batch_size),          
                        max_queue_size=1,
                        epochs=args.epochs,
                        steps_per_epoch=args.steps_per_epochs,
                        validation_data=batch_generator(valFile, batch_size),
                        validation_steps=len(valFile)//batch_size,
                        callbacks=[checkpoint]
                       )


    

def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-w', '--weights',
        metavar='PATH',
        dest='weights',
        default=None,
        help='Path to existing weights file')
    argparser.add_argument(
        '-b', '--batch_size',
        dest='batch_size',
        default=120,
        help='Number of training examples utilised in one iteration')
    argparser.add_argument(
        '-e', '--epochs',
        dest='epochs',
        default=5,
        help='Number of epochs')
    argparser.add_argument(
        '-s', '--steps',
        dest='steps_per_epochs',
        default=500,
        help='Number of steps per epochs')

    args = argparser.parse_args()

    train_CIL(args)
    print('Training...')


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user.')