from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

num_classes = 10


base_model = InceptionV3(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=["accuracy"])

# train the model on the new data for a few epochs

img_width, img_height = 299, 299
train_data_dir = "/tmp/food-101-10/train"
validation_data_dir = "/tmp/food-101-10/test"
nb_train_samples = 7010
nb_validation_samples = 1010
batch_size = 512
epochs = 25

# Initiate the train and test generators with data Augumentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.3,
    horizontal_flip=True,
    fill_mode="nearest",
    zoom_range=0.3,
    width_shift_range=0.9,
    height_shift_range=0.9,
    rotation_range=70)

test_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.3,
    horizontal_flip=True,
    fill_mode="nearest",
    zoom_range=0.3,
    width_shift_range=0.9,
    height_shift_range=0.9,
    rotation_range=70)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical")

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    class_mode="categorical")

# Save the model according to the conditions
checkpoint = ModelCheckpoint("inception_v3.h5", monitor='val_acc', verbose=2, save_best_only=True,
                             save_weights_only=False,
                             mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

# steps_per_epoch: Total number of steps (batches of samples) to yield from generator before declaring one epoch finished and starting the next epoch.
# It should typically be equal to the number of unique samples of your dataset divided by the batch size.

# validation_steps: Only relevant if validation_data is a generator. Number of steps to yield from validation generator at the end of every epoch.
# It should typically be equal to the number of unique samples of your validation dataset divided by the batch size.

# Train the model
# model.fit_generator(
#     train_generator,
#     steps_per_epoch=nb_train_samples/batch_size,
#     validation_steps=nb_validation_samples/batch_size,
#     epochs=epochs,
#     use_multiprocessing=True,
#     validation_data=validation_generator,
#     callbacks=[checkpoint, early])
#



# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
    print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:240]:
    layer.trainable = False
for layer in model.layers[240:]:
    layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD

model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy', metrics=["accuracy"])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers




checkpoint = ModelCheckpoint("inception_v3.h5", monitor='val_acc', verbose=1, save_best_only=True,
                             save_weights_only=False,
                             mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

# Train the model
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples / batch_size,
    validation_steps=nb_validation_samples / batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[checkpoint, early])
