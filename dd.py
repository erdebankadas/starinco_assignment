import pandas as pd
import numpy as np
# import os
# import ast
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

georges_data = pd.read_csv(r'E:\DD_Duo\starincoo\georges.csv')# paste georges path
non_georges_data = pd.read_csv(r'E:\DD_Duo\starincoo\non_georges.csv')#paste non georges path

georges_data['target'] = 1
non_georges_data['target'] = 0

georges_data.columns = ['X', 'Y']

georges_data.head()

non_georges_data.columns = ['X', 'Y']

non_georges_data.head()

georges_data.shape

non_georges_data.shape

df = pd.concat([georges_data, non_georges_data], ignore_index=True)

# Shuffle (randomly reorder) the rows
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.shape

base_model = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# Add custom classification layers
x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

from PIL import Image
import requests
from io import BytesIO
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def custom_data_generator(df, image_size, batch_size):
    while True:
        # Initialize empty lists to store batch data
        batch_X = []
        batch_Y = []

        # Randomly select 'batch_size' indices from the DataFrame
        indices = np.random.choice(df.index, size=batch_size, replace=False)

        for index in indices:
            row = df.loc[index]
            image_url = row['X']
            label = row['Y']

            # Download and preprocess the image
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))

            # Resize the image while maintaining the aspect ratio
            image = image.resize(image_size)

            # Convert the image to RGB mode (in case it's grayscale)
            image = image.convert('RGB')

            # Convert to numpy array and normalize pixel values
            image = np.array(image) / 255.0

            batch_X.append(image)
            batch_Y.append(label)

        yield (np.array(batch_X), np.array(batch_Y))


# Create custom data generators
image_size = (224, 224)
batch_size = 32

train_generator = custom_data_generator(df, image_size, batch_size)
test_generator = custom_data_generator(df, image_size, batch_size)

history = model.fit(train_generator, epochs=10, validation_data=test_generator)# if needed speedup then change epochs =10 to 5

test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

from tensorflow.keras.preprocessing import image
import numpy as np

# Load and preprocess the image
img = image.load_img(r'E:\DD_Duo\starincoo\george_test_task\no_george\0a1a0665ac98ae81be4b54db17564f34.jpg', target_size=(224, 224)) # paste your image path
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = img / 255.0  # Rescale pixel values

# Make a prediction
prediction = model.predict(img)

if prediction[0] > 0.5:
    print("St. George is present in the image.")
else:
    print("St. George is not present in the image.")