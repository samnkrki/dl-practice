
'''
https://stackoverflow.com/questions/66714485/how-can-i-train-my-cnn-model-on-dataset-from-a-csv-file
https://datatofish.com/list-to-dataframe/

'''
import tensorflow as tf
from glob import glob
from sklearn.model_selection import train_test_split
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
import pandas as pd

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.gridspec as gridspec

def get_dataset(dataset_root):
    train_folder = [];
    test_folder = [];
    total_classes = 0;
    for root,dirs,files in os.walk(dataset_root):
        total_classes = total_classes + len(dirs)
        for each_dir in dirs:
            path_to_class = glob(os.path.join(root, each_dir) + '/*.jpg')
            train, test = train_test_split(path_to_class, test_size=0.30) # train, test path is captured, equal split from each folder
            train_folder.append(train)
            test_folder.append(test)

    return train_folder, test_folder, total_classes

#flatten array of items from each folder to a single train array
def flatten_train_test():
    train_items, test_items, total_classes =  get_dataset('/home/samin/Documents/datasets/catsAndDogs/PetImages')
    return [item for sublist in train_items for item in sublist], [item for sublist in test_items for item in sublist], total_classes
    
train_data, test_data, total_class = flatten_train_test()

def convert_to_df(data):
    obj = {
        'filepath': data,
        'label': [datum.split('/')[-2] for datum in data]
        }
    return pd.DataFrame(obj, columns = ['filepath','label'])

test_df = convert_to_df(test_data)
train_df = convert_to_df(train_data)
#take random items

def get_random_sample(total_list, sample_num=1):
    return np.random.choice(train_data, size=sample_num, replace=False, p=None)

plotting_items = get_random_sample(train_data, 25);

print(plotting_items)

def plot_images(plot_list, row, col):
    plt.figure(figsize=(12,9))
    for (i,each_img_path) in enumerate(plot_list):
        label = each_img_path.split("/")[-2]
        #print(each_img_path)
        im = Image.open(each_img_path).convert('RGB')
        plt.subplot(row, col, i+1)
        plt.title(label)
        plt.imshow(np.asarray(im))
        plt.axis('off')
        
plot_images(plotting_items, 5,5)


#creating a model
base_model = InceptionV3(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D(name='avg_pool')(x)
x = Dropout(0.4)(x)
preds = Dense(total_class, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs = preds)


#add transfer learning
for layer in base_model.layers:
    layer.trainable = False
    
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

IMG_WIDTH = 224
IMG_HEIGHT=224
BATCH_SIZE=320
EPOCHS=5
IMG_SIZE=(IMG_WIDTH, IMG_HEIGHT)

#data prepration
train_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.inception_v3.preprocess_input, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
val_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.inception_v3.preprocess_input, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

train_generator = train_datagen.flow_from_dataframe(train_df, x_col='filepath', y_col='label', class_mode='categorical', batch_size=BATCH_SIZE, target_size=(IMG_HEIGHT, IMG_WIDTH))
validation_generator = train_datagen.flow_from_dataframe(test_df, x_col='filepath', y_col='label', class_mode='categorical', batch_size=BATCH_SIZE, target_size=(IMG_HEIGHT, IMG_WIDTH))

STEPS_PER_EPOCH=32
VALIDATION_STEPS=6
MODEL_FILE = 'outputs/cat_dog_classification.model'

history = model.fit(train_generator, epochs=EPOCHS, validation_steps=6)

model.save(MODEL_FILE)
#plot training
def plot_training(history):
    acc = history.history['accuracy']
    #val_acc = history.history['val_acc']
    loss = history.history['loss']
    #val_loss = history.history['val_loss']
    epochs = range(len(acc))
    
    plt.plot(epochs, acc, 'r.')
    #plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation loss')
    plt.show()

plot_training(history)
print(history.history)
#prediction 

def predict(model, img):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return preds[0]

def plot_pred(img, preds):
    labels = ["cat", "dog"]
    gs = gridspec.GridSpec(2, 1, height_ratios=[4,1])
    plt.figure(figsize=(8,8))
    plt.subplot(gs[0])
    plt.imshow(np.asarray(img))
    plt.subplot(gs[1])
    plt.barh([0,1], preds, alpha=0.5)
    plt.yticks([0,1], labels)
    plt.xlabel('Probability')
    plt.xlim(0,1)
    plt.tight_layout()
    
model = load_model(MODEL_FILE)

pred_src='assets/lion.jpg'


img = image.load_img(pred_src, target_size=IMG_SIZE)
preds = predict(model, img)
plot_pred(np.asarray(img), preds)
preds
'''        '''