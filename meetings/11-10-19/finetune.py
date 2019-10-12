import sys

from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

num_classes = 4
epochs = 3
batch_size = 128
input_shape = (224, 224, 3)
train_steps = 2000 // batch_size
test_steps = 1000 // batch_size
try:
    data_augmentation = bool(sys.argv[1])
except IndexError:
    data_augmentation = False

# Setup generator
if not data_augmentation:
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
else:
    train_datagen = ImageDataGenerator(rescale=(1.0 / 255))

test_datagen = ImageDataGenerator(rescale=(1.0 / 255))

train_generator = train_datagen.flow_from_directory(
    "data/train", target_size=input_shape[:-1], batch_size=batch_size
)

validation_generator = test_datagen.flow_from_directory(
    "data/test", target_size=input_shape[:-1], batch_size=batch_size
)

# Model

base = ResNet50V2(include_top=False, pooling="avg", input_shape=input_shape)
x = Dense(num_classes)(base.output)
x = Activation("softmax", name="Prediction")(x)
model = Model(inputs=base.input, outputs=x)

for layer in model.layers:
    layer.trainable = False
model.layers[-1].trainable = True
model.layers[-2].trainable = True

print(model.summary())
print([layer.trainable for layer in model.layers])

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_steps,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=test_steps,
)
