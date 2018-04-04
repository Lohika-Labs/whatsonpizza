from PIL import ImageFile
from keras import optimizers
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

ImageFile.LOAD_TRUNCATED_IMAGES = True

num_classes = 21

img_width, img_height = 299, 299
train_data_dir = "/mnt/data/lab/datasets/pizza/pizza labeled/small/augmented_train"
validation_data_dir = "/mnt/data/lab/datasets/pizza/pizza labeled/small/test"
nb_train_samples = 28665
nb_validation_samples = 910
batch_size = 256
epochs = 150

base_model = InceptionV3(weights='imagenet', include_top=False, input_shape = (img_width, img_height, 3))

for layer in base_model.layers:
    layer.trainable = False


x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(loss="categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    # shear_range=0.3,
    # horizontal_flip=True,
    # fill_mode="nearest",
    # zoom_range=0.3,
    # width_shift_range=0.3,
    # height_shift_range=0.3,
    # rotation_range=70

)

test_datagen = ImageDataGenerator(
    rescale=1. / 255,
    # shear_range=0.3,
    # horizontal_flip=True,
    # fill_mode="nearest",
    # zoom_range=0.3,
    # width_shift_range=0.3,
    # height_shift_range=0.3,
    # rotation_range=70
)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical")

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size = 256,
    class_mode="categorical")


checkpoint = ModelCheckpoint("inception_v3.h5", monitor='val_acc', verbose=2, save_best_only=True,
                             save_weights_only=False,
                             mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=50, verbose=1, mode='auto')


#csv_logger = CSVLogger('model4.log')


model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples/batch_size,
    validation_steps=nb_validation_samples/batch_size,
    epochs=epochs,
    use_multiprocessing=True,
    validation_data=validation_generator,
    callbacks=[checkpoint, early])



