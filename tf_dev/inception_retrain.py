import argparse
import os
import os.path

from PIL import ImageFile
from keras import optimizers
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, TensorBoard
from keras.layers import Dense, Dropout, Flatten, AveragePooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras_sequential_ascii import keras2ascii

ImageFile.LOAD_TRUNCATED_IMAGES = True


'''
  set global configuratin
  num_classes - total number of classes in dataset. In our case we have 10
  batch_size  - how many images we process in single batch
  epochs - number of iterations over whole dataset
  
  
'''
num_classes = 10
batch_size = 16
epochs = 500


def get_num_of_files(root_dir):
    total = 0
    for root, dirs, files in os.walk(root_dir):
        total += len(files)
    return total


def retrain(train_data_dir, validation_data_dir):
    # image size for input layer
    img_width, img_height = 299, 299

    #get number of samples in train dir
    nb_train_samples = get_num_of_files(train_data_dir)
    #get number of samples in validation dir
    nb_validation_samples = get_num_of_files(validation_data_dir)

    # we are going to use InceptionV3 pretrained model and apply transfer learning approach
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

    x = base_model.output

    #create and initialize output layer
    x = Dense(128, activation='relu', init='glorot_uniform', W_regularizer=l2(.0005))(x)
    x = Dropout(0.7)(x)

    x = AveragePooling2D(pool_size=(8, 8))(x)
    x = Dropout(.7)(x)
    x = Flatten()(x)
    predictions = Dense(num_classes, init='glorot_uniform', W_regularizer=l2(.0005), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    #print model summary
    print(keras2ascii(model))


    opt = optimizers.SGD(lr=.01, momentum=.9)

    # define learning rate decay strategy
    def schedule(epoch):
        if epoch < 20:
            return 0.01
        if epoch < 35:
            return 0.001
        elif epoch < 1000:
            return .0005

    lr_scheduler = LearningRateScheduler(schedule)

    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    #create train image generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.3,
        horizontal_flip=True,
        fill_mode="nearest",
        zoom_range=0.3,
        width_shift_range=0.3,
        height_shift_range=0.3,
        rotation_range=70
    )

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode="categorical")

    #create train image generator
    test_datagen = ImageDataGenerator(
        rescale=1. / 255,
    )

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=256,
        class_mode="categorical")

    #define checkpoint, so our model would be saved if conditions are met (best models only)
    checkpoint = ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.hdf5', monitor='val_acc',
                                 verbose=2, save_best_only=True,
                                 save_weights_only=False,
                                 mode='auto', period=1)

    #stop training if there are no improvements after 50 epochs
    early = EarlyStopping(monitor='val_acc', min_delta=0, patience=50, verbose=1, mode='auto')

    #write logs for tensorboard
    tensorboard = TensorBoard(log_dir='/tmp/tb_logs/metrics', histogram_freq=2, write_graph=True, write_images=False)
    tensorboard.set_model(model)

    #start training
    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples / batch_size,
        validation_steps=nb_validation_samples / batch_size,
        epochs=epochs,
        use_multiprocessing=True,
        validation_data=validation_generator,
        callbacks=[lr_scheduler, early, checkpoint, tensorboard])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("train")
    parser.add_argument("test")

    args = parser.parse_args()
    retrain(args.train, args.test)
