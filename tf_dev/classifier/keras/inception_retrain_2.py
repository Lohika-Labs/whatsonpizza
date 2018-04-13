from PIL import ImageFile
from keras import optimizers
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, LearningRateScheduler
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten, AveragePooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2

ImageFile.LOAD_TRUNCATED_IMAGES = True

num_classes = 10

img_width, img_height = 299, 299
train_data_dir = "/mnt/data/lab/datasets/pizza/pizza labeled/cleaned/train"
validation_data_dir = "/mnt/data/lab/datasets/pizza/pizza labeled/cleaned/test"
nb_train_samples = 6104*4
nb_validation_samples = 382
batch_size = 64
epochs = 150

base_model = InceptionV3(weights='imagenet', include_top=False, input_shape = (img_width, img_height, 3))

for layer in base_model.layers:
    layer.trainable = True


x = base_model.output

x = Dense(128, activation='relu',  init='glorot_uniform', W_regularizer=l2(.0005))(x)
x = Dropout(0.8)(x)

x = AveragePooling2D(pool_size=(8, 8))(x)
x = Dropout(.7)(x)
x = Flatten()(x)
predictions = Dense(num_classes, init='glorot_uniform', W_regularizer=l2(.0005), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

opt = optimizers.SGD(lr=.01, momentum=.9)


def schedule(epoch):
    if epoch < 10:
        return 0.01
    if epoch < 20:
        return 0.001
    elif epoch < 50:
        return .0005

lr_scheduler = LearningRateScheduler(schedule)


model.compile(loss="categorical_crossentropy", optimizer = opt, metrics=["accuracy"])


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

test_datagen = ImageDataGenerator(
    rescale=1. / 255,
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


checkpoint = ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.hdf5', monitor='val_acc', verbose=2, save_best_only=True,
                             save_weights_only=False,
                             mode='auto', period=1)


early = EarlyStopping(monitor='val_acc', min_delta=0, patience=50, verbose=1, mode='auto')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples/batch_size,
    validation_steps=nb_validation_samples/batch_size,
    epochs=epochs,
    use_multiprocessing=True,
    validation_data=validation_generator,
    callbacks=[lr_scheduler, early, checkpoint])