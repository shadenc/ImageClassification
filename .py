import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img,array_to_img
import numpy as np
import matplotlib.pyplot as plt
dataset_path = '/kaggle/input/garbage-classification/Garbage classification/Garbage classification'
batch_size = 32
img_height = 224
img_width = 224
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    validation_split=0.3,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    validation_split=0.3,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)


class_names = train_ds.class_names
Found 2527 files belonging to 6 classes.
Using 1769 files for training.
Found 2527 files belonging to 6 classes.
Using 758 files for validation.
dataset=[]
testset=[]
count=0
for file in os.listdir(directory):
    path=os.path.join(directory,file)
    t=0
    for im in os.listdir(path):
        image=load_img(os.path.join(path,im), grayscale=False, color_mode='rgb', target_size=(100,100))
        image=img_to_array(image)
        image=image/255.0
        t+=1
        if t<=300:
            dataset.append([image,count])
        else:
            testset.append([image,count])        
    count=count+1 class_names
new_dataset_base_dir = '/kaggle/working/newdataset'

for class_name in class_names:
    new_class_dir = os.path.join(new_dataset_base_dir, class_name)
    os.makedirs(new_class_dir, exist_ok=True)
import shutil

for class_name in class_names:
    original_class_dir = os.path.join(dataset_path, class_name)
    new_class_dir = os.path.join(new_dataset_base_dir, class_name)

    for filename in os.listdir(original_class_dir):
        src_path = os.path.join(original_class_dir, filename)
        dst_path = os.path.join(new_class_dir, filename)
        shutil.copy(src_path, dst_path)
target_class_counts = {'cardboard': 191, 'glass': 93, 'metal': 184, 'paper': 0, 'plastic': 112, 'trash': 457}
from tensorflow.keras.preprocessing.image import ImageDataGenerator

 
data_gen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


for class_name, additional_images_needed in target_class_counts.items():
    if additional_images_needed > 0:
        original_class_dir = os.path.join(dataset_path, class_name)
        new_class_dir = os.path.join(new_dataset_base_dir, class_name)   
        image_files = os.listdir(original_class_dir)
        augmented_images_count = 0

        for image_file in image_files:
            if augmented_images_count >= additional_images_needed:
                break   

            img_path = os.path.join(original_class_dir, image_file)
            img = load_img(img_path)  
            img_array = img_to_array(img)   
            img_array = np.expand_dims(img_array, axis=0)   

 
            for _ in data_gen.flow(img_array, batch_size=1,
                                   save_to_dir=new_class_dir, 
                                   save_prefix=f"aug_{class_name}", 
                                   save_format='jpeg'):
                augmented_images_count += 1
                if augmented_images_count >= additional_images_needed:
                    break
class_dir = '/kaggle/working/newdataset/trash'
file_count = len([name for name in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, name))])
file_count
dataset_path = '/kaggle/working/newdataset'
 
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  dataset_path,
  validation_split=0.3,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  dataset_path,
  validation_split=0.3,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Normalization
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
from tensorflow.keras import models, layers

# Load ResNet50 model
base_model = tf.keras.applications.ResNet50(input_shape=(img_height, img_width, 3),
                                               include_top=False,  
                                               weights='imagenet')
base_model.trainable = False
base_model.trainable = False

model = models.Sequential([
  base_model,
  layers.GlobalAveragePooling2D(),
  layers.Dense(6, activation='softmax')  
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# Unfreeze 
base_model.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# (fine-tuning)
fine_tune_epochs = 10
total_epochs = epochs + fine_tune_epochs

history_fine = model.fit(train_ds,
                         validation_data=val_ds,
                         epochs=total_epochs,
                         initial_epoch=history.epoch[-1])
from tensorflow.keras.preprocessing import image
import numpy as np

img_path = '/kaggle/input/testaa/box.webp' 
img = image.load_img(img_path, target_size=(224, 224)) 
img_array = image.img_to_array(img)   
img_array = np.expand_dims(img_array, axis=0)   
img_array /= 255.0  


predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)


predicted_class_name = class_names[predicted_class[0]]
print(f"Model prediction: {predicted_class_name}")
 
get_acc = his.history['accuracy']
value_acc = his.history['val_accuracy']
get_loss = his.history['loss']
validation_loss = his.history['val_loss']

epochs = range(len(get_acc))
plt.plot(epochs, get_acc, 'r', label='Accuracy of Training data')
plt.plot(epochs, value_acc, 'b', label='Accuracy of Validation data')
plt.title('Training vs validation accuracy')
plt.legend(loc=0)
plt.figure()
plt.show()
