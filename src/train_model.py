import util
import numpy as np
import os
import gc

IMG_SIZE = 15


def generate_model():
    from keras.layers import Input, Conv2D, Reshape
    from keras.layers import Activation, BatchNormalization
    from keras.models import Model
    from keras.optimizers import Adam
    import keras

    input_boards = Input(shape=(15, 15, 3))
    x = Conv2D(16, (3, 3), padding='same')(input_boards)
    x = BatchNormalization()(Activation('relu')(x))
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(Activation('relu')(x))
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(Activation('relu')(x))

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(Activation('relu')(x))
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(Activation('relu')(x))
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(Activation('relu')(x))

    x = Conv2D(1, (1, 1), padding='same')(x)

    predictions = Activation('softmax')(Reshape((225,))(x))
    model = Model(inputs=input_boards,
                       outputs=predictions)

    model.compile(optimizer=keras.optimizers.Adam(),
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])

    model.summary()

    return model


def generate_train(file_name):
    file = open(file_name)
    states = []
    labels = []
    for line in file.readlines():
        res, *moves = line.split()
        field = np.zeros((15, 15, 3), dtype=np.int8)
        field[:, :, 2] = 1
        for move in moves:
            states.append(field)
            cur_move = util.to_pos(move)
            labels.append(cur_move[0] * 15 + cur_move[1])
            field[cur_move[0], cur_move[1], 0] = 1
            field[:, :, [0, 1]] = field[:, :, [1, 0]]
            field[:, :, 2] = 0 ** field[0, 0, 2]
    return np.array(states, dtype=np.int8), np.array(labels)


os.system('split -C 27m --numeric-suffixes train-1.renju data')
print('start parsing')
for i in range(9):
    train_x, train_y = generate_train('data0' + str(i))
    np.save('parsed_x' + str(i), train_x)
    np.save('parsed_y' + str(i), train_y)
    del train_x, train_y
    print('data generated', i + 1, 'out of', 9)

model = generate_model()
print('input count of epochs')
epochs_count = int(input())
for i in range(epochs_count):
    for j in range(9):
        print('started epoch', i + 1)
        train_x = np.load('parsed_x' + str(j) + '.npy')
        train_y = np.load('parsed_y' + str(j) + '.npy')
        model.fit(train_x, train_y, 300, 1)
        del train_x
        del train_y
        gc.collect()

model.save('your_model')
for i in range(9):
    os.remove('data' + str(i))
    os.remove('parsed_x' + str(i) + '.npy')
    os.remove('parsed_y' + str(i) + '.npy')

