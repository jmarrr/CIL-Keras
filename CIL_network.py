from __future__ import print_function
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, concatenate


def CIL(input_image, input_speed, weights_path = None):
    """
       Parameters
       ----------
            input_image : tuple
                Image input shape.
                
            input_speed : tuple
                Speed measurements input shape.
                
            masks : list (not available)
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

    
    print(branches)
    
    # Load weights
    if weights_path:
        model.load_weights(weights_path)
        
    return model
    