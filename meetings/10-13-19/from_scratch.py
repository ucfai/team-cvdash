import sys

from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

num_classes = 4
epochs = 3
batch_size = 32
input_shape = (224, 224, 3)
train_steps = 2000 // batch_size
test_steps = 1000 // batch_size
try:
    data_augmentation = sys.argv[1]
except IndexError:
    data_augmentation = False


model = ResNet50V2(weights=None, classes=num_classes, input_shape=input_shape)

if data_augmentation:
    train_datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        preprocessing_function=preprocess_input
    )
else:
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)


train_generator = train_datagen.flow_from_directory(
    "data/train", target_size=input_shape[:-1], batch_size=batch_size
)

validation_generator = test_datagen.flow_from_directory(
    "data/test", target_size=input_shape[:-1], batch_size=batch_size
)

model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

print(model.summary())

history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_steps,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=test_steps,
)
