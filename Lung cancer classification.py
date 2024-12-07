#!/usr/bin/env python
# coding: utf-8

# In[29]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# In[30]:


base_path = '/Users/vamshisamudrala/Downloads/lung_colon_image_set'


# In[31]:


IMG_SIZE = 150

X = []
Z = []


# In[32]:


def assign_label(img, category):
    return category

def make_train_data(category, DIR):
    for img in os.listdir(DIR):
        label = assign_label(img, category)
        path = os.path.join(DIR, img)
        img_data = cv2.imread(path, cv2.IMREAD_COLOR)
        
        if img_data is not None:
            img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
            X.append(np.array(img_data))  
            Z.append(str(label)) 
        else:
            print(f"Error reading image: {path}")


# In[33]:


COLON_ACA_DIR = os.path.join(base_path, 'colon_aca')
COLON_N_DIR = os.path.join(base_path,'colon_n')
LUNG_ACA_DIR = os.path.join(base_path, 'lung_aca')
LUNG_N_DIR = os.path.join(base_path, 'lung_n')
LUNG_SCC_DIR = os.path.join(base_path, 'lung_scc')


# In[34]:


make_train_data('colon_aca', COLON_ACA_DIR)
make_train_data('colon_n', COLON_N_DIR)
make_train_data('lung_aca', LUNG_ACA_DIR)
make_train_data('lung_n', LUNG_N_DIR)
make_train_data('lung_scc', LUNG_SCC_DIR)


# In[35]:


print(f"Total images collected: {len(X)}")
print(f"Total labels collected: {len(Z)}")


# In[36]:


categories = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

def show_sample_images(base_path, categories, num_samples=5):
    fig, axes = plt.subplots(len(categories), num_samples, figsize=(12, 8))
    for i, category in enumerate(categories):
        folder_path = os.path.join(base_path, category)
        images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))][:num_samples]
        for j, img_name in enumerate(images):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[i, j].imshow(img)
                axes[i, j].axis('off')
                if j == 0:
                    axes[i, j].set_title(category)
    plt.tight_layout()
    plt.show()

show_sample_images(base_path, categories)


# In[37]:


category_counts = {
    'colon_aca': len(os.listdir(os.path.join(base_path, 'colon_aca'))),
    'colon_n': len(os.listdir(os.path.join(base_path, 'colon_n'))),
    'lung_aca': len(os.listdir(os.path.join(base_path, 'lung_aca'))),
    'lung_n': len(os.listdir(os.path.join(base_path, 'lung_n'))),
    'lung_scc': len(os.listdir(os.path.join(base_path, 'lung_scc')))
}

plt.figure(figsize=(10, 6))
sns.barplot(x=list(category_counts.keys()), y=list(category_counts.values()))
plt.title('Image Count per Category')
plt.ylabel('Number of Images')
plt.xlabel('Category')
plt.show()


# In[13]:


import random
import gc

def plot_pixel_intensity_distribution(base_path, categories, max_images_per_category=100):
    plt.figure(figsize=(12, 6))
    
    for category in categories:
        folder_path = os.path.join(base_path, category)
        img_names = os.listdir(folder_path)
        pixel_values = []

        sampled_images = random.sample(img_names, min(max_images_per_category, len(img_names)))

        for img_name in sampled_images:
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                pixel_values.extend(img.flatten())

        sns.histplot(pixel_values, bins=50, kde=True, label=category, alpha=0.6)
        
        del pixel_values
        gc.collect()

    plt.title('Pixel Intensity Distribution Across Categories')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

plot_pixel_intensity_distribution(base_path, categories)


# In[13]:


selected_categories = ['lung_scc', 'lung_aca']

def plot_pixel_intensity_distribution(base_path, categories, max_images_per_category=100):
    plt.figure(figsize=(12, 6))
    
    for category in categories:
        folder_path = os.path.join(base_path, category)
        img_names = os.listdir(folder_path)
        pixel_values = []

        sampled_images = random.sample(img_names, min(max_images_per_category, len(img_names)))

        for img_name in sampled_images:
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                pixel_values.extend(img.flatten())

        sns.histplot(pixel_values, bins=50, kde=True, label=category, alpha=0.6)
        
        del pixel_values
        gc.collect()

    plt.title('Pixel Intensity Distribution for Lung SCC and Lung ACA')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

plot_pixel_intensity_distribution(base_path, selected_categories)


# In[38]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

le = LabelEncoder()
Y = le.fit_transform(Z)
Y = to_categorical(Y, len(categories))

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

print(f"x_train shape: {np.array(x_train).shape}")
print(f"y_train shape: {np.array(y_train).shape}")
print(f"x_test shape: {np.array(x_test).shape}")
print(f"y_test shape: {np.array(y_test).shape}")


# **Pretrained Model**

# In[12]:


import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

#Freeze
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(512, activation='relu')(x)
output = tf.keras.layers.Dense(5, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)


# In[13]:


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import CategoricalAccuracy

x_train = np.array(x_train)
x_train = x_train.reshape(x_train.shape[0], IMG_SIZE, IMG_SIZE, 3)

x_test = np.array(x_test)
x_test = x_test.reshape(y_test.shape[0], IMG_SIZE, IMG_SIZE, 3)

#Compile
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=[CategoricalAccuracy()])

epochs = 10
batch_size = 64
history = model.fit(x_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(x_test, y_test),
                    verbose=1)


# In[14]:


import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

#accuracy
ax1.plot(history.history['categorical_accuracy'], label='Train Accuracy', marker='o')
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend(loc='lower right')
ax1.set_xticks(range(0, 10)) 
ax1.set_xticklabels(range(1, 11)) 
ax1.grid(True)

#loss
ax2.plot(history.history['loss'], label='Train Loss', marker='o')
ax2.set_title('Model Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend(loc='upper right')
ax2.set_xticks(range(0, 10))  
ax2.set_xticklabels(range(1, 11))  
ax2.grid(True)

plt.tight_layout()
plt.show()


# In[15]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.models import Model
import numpy as np

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

#metrics
accuracy = accuracy_score(y_true_classes, y_pred_classes) * 100
precision = precision_score(y_true_classes, y_pred_classes, average='macro') *100
recall = recall_score(y_true_classes, y_pred_classes, average='macro') * 100
f1 = f1_score(y_true_classes, y_pred_classes, average='macro') * 100
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)


#metrics
print(f"Accuracy: {accuracy:.2f}%")
print(f"Precision: {precision:.2f}%")
print(f"Recall: {recall:.2f}%")
print(f"F1 Score: {f1:.2f}%")


# In[16]:


classes = ["colon_aca", "colon_n", "lung_aca", "lung_n", "lung_scc"]

plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()


# **CNN Model**

# In[18]:


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,Dropout


# In[19]:


x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

x_train = x_train / 255.0
x_test = x_test / 255.0


# In[20]:


model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

# Dropout
model.add(Dropout(0.5))

model.add(Dense(5, activation='softmax'))


# In[21]:


from tensorflow.keras.optimizers import Adam

optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


# In[22]:


history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=10, 
    batch_size=32
)

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')


# In[23]:


plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()


# In[24]:


y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)


# In[25]:


from sklearn.metrics import precision_score, recall_score, accuracy_score

precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
accuracy = accuracy_score(y_true_classes, y_pred_classes)

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'Accuracy: {accuracy:.4f}')


# In[26]:


from sklearn.metrics import confusion_matrix

y_pred = model.predict(x_test)

y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true_classes, y_pred_classes)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


# **Fine Tuning CNN Model**

# In[28]:


from tensorflow.keras.optimizers import Adam

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


# In[29]:


history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=15,
    batch_size=32
)

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')


# In[30]:


# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()


# In[31]:


y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)


# In[32]:


from sklearn.metrics import precision_score, recall_score, accuracy_score

#precision, recall, and accuracy
precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
accuracy = accuracy_score(y_true_classes, y_pred_classes)

#metrics
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'Accuracy: {accuracy:.4f}')


# In[33]:


from sklearn.metrics import confusion_matrix

y_pred = model.predict(x_test)

y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true_classes, y_pred_classes)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


# **CSIP(Class-Specific Image Processing)**

# In[2]:


import os
import cv2

base_path = '/Users/vamshisamudrala/Downloads/lung_colon_image_set'
processed_base_path = '/Users/vamshisamudrala/Desktop/lung_colon_image_set_processed'

os.makedirs(processed_base_path, exist_ok=True)


# In[3]:


def enhance_contrast(image):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV) 
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])  
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR) 


# In[4]:


target_classes = ['lung_scc', 'lung_aca']

for category in target_classes:
    category_path = os.path.join(base_path, category)
    processed_category_path = os.path.join(processed_base_path, category)
    os.makedirs(processed_category_path, exist_ok=True)
    
    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)
        
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        
        if img is not None:
            enhanced_img = enhance_contrast(img)
  
            enhanced_img_path = os.path.join(processed_category_path, img_name)
            cv2.imwrite(enhanced_img_path, enhanced_img)
        else:
            print(f"Error reading image: {img_path}")


# **Pre-defined model with Class-Specific Image Processing**

# In[7]:


import os
import numpy as np
import cv2
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


# In[8]:


original_data_path = '/Users/vamshisamudrala/Downloads/lung_colon_image_set'
processed_data_path = '/Users/vamshisamudrala/Desktop/lung_colon_image_set_processed'


# In[9]:


categories = ['colon_aca', 'colon_n', 'lung_aca', 'lung_n', 'lung_scc']

IMG_SIZE = (150, 150)


# In[10]:


def load_images(categories, original_path, processed_path, img_size=(150, 150)):
    X, y = [], []
    for category in categories:
        if category in ['lung_scc', 'lung_aca']:
            folder_path = os.path.join(processed_path, category)
        else:
            folder_path = os.path.join(original_path, category)
        
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, img_size)
                X.append(img)
                y.append(categories.index(category)) 

    return np.array(X), np.array(y)


# In[11]:


X, y = load_images(categories, original_data_path, processed_data_path, IMG_SIZE)

y = to_categorical(y, num_classes=len(categories))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data loaded and split successfully.")
print(f"Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}")
print(f"Testing data shape: {X_test.shape}, Testing labels shape: {y_test.shape}")


# In[12]:


base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
output = Dense(len(categories), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)


# In[13]:


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")


# In[14]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()


# In[15]:


from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[17]:


y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1) 
class_names = ["colon_aca", "colon_n", "lung_aca", "lung_n", "lung_scc"]

print(classification_report(y_true, y_pred, target_names=class_names))


# In[18]:


conf_matrix = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# **CSIP CNN Model**

# In[3]:


import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

base_path = '/Users/vamshisamudrala/Downloads/lung_colon_image_set'
processed_path = '/Users/vamshisamudrala/Desktop/lung_colon_image_set_processed'
IMG_SIZE = 150  

X = []
Z = []


# In[4]:


def load_images_from_folder(folder, label):
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img)
            Z.append(label)


# In[5]:


load_images_from_folder(os.path.join(base_path, 'colon_aca'), 0) 
load_images_from_folder(os.path.join(base_path, 'colon_n'), 1)   
load_images_from_folder(os.path.join(base_path, 'lung_n'), 2)     

load_images_from_folder(os.path.join(processed_path, 'lung_aca'), 3)  
load_images_from_folder(os.path.join(processed_path, 'lung_scc'), 4)


# In[ ]:


x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

x_train = x_train / 255.0
x_test = x_test / 255.0


# In[4]:


X = np.array(X) / 255.0
Z = np.array(Z)

Z = to_categorical(Z, num_classes=5)

X_train, X_val, y_train, y_val = train_test_split(X, Z, test_size=0.2, random_state=42)


# In[5]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential()

# Firstlayer
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Secondlayer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Thirdlayer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatteninglayers
model.add(Flatten())

# Fullylayer
model.add(Dense(128, activation='relu'))

# Dropout
model.add(Dropout(0.5))

# Output layer
model.add(Dense(5, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[7]:


history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=32)

val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {val_accuracy*100:.2f}%")


# In[8]:


from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_val, axis=1)


# In[9]:


import seaborn as sns
import matplotlib.pyplot as plt

class_names = ["colon_aca", "colon_n", "lung_aca", "lung_n", "lung_scc"]
print(classification_report(y_true, y_pred_classes, target_names=class_names))


# In[10]:


conf_matrix = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


# **without hot encoding**

# In[2]:


import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

base_path = '/Users/vamshisamudrala/Downloads/lung_colon_image_set'
processed_path = '/Users/vamshisamudrala/Desktop/lung_colon_image_set_processed'
IMG_SIZE = 150 

X = []
Z = []


# In[3]:


for label, class_name in enumerate(os.listdir(base_path)):
    class_path = os.path.join(base_path, class_name)
    if os.path.isdir(class_path):  
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  
            X.append(img) 
            Z.append(label)  


# In[4]:


X = np.array(X)
Z = np.array(Z)

X = X / 255.0

X_train, X_val, y_train, y_val = train_test_split(X, Z, test_size=0.2, random_state=42)


# In[5]:


print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("Unique labels in y_train:", np.unique(y_train))

print("Check for NaNs in X_train:", np.isnan(X_train).any())
print("Check for NaNs in y_train:", np.isnan(y_train).any())


# In[6]:


model = Sequential()

# Firstlayer
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Secondlayer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Thirdlayer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatteninglayers
model.add(Flatten())

# Fully connected layer
model.add(Dense(128, activation='relu'))

# Dropout
model.add(Dropout(0.5))

# Output layer
model.add(Dense(5, activation='softmax'))


# In[7]:


optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[8]:


y_train_adjusted = y_train - 1
y_val_adjusted = y_val - 1

history = model.fit(X_train, y_train_adjusted, epochs=10, validation_data=(X_val, y_val_adjusted), batch_size=32)

val_loss, val_accuracy = model.evaluate(X_val, y_val_adjusted)
print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")


# In[20]:


import matplotlib.pyplot as plt

train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
epochs = range(1, len(train_loss) + 1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, label='Training Loss', color='blue')
plt.plot(epochs, val_loss, label='Validation Loss', color='orange')
plt.title('Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracy, label='Training Accuracy', color='blue')
plt.plot(epochs, val_accuracy, label='Validation Accuracy', color='orange')
plt.title('Accuracy vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()


# In[19]:


import numpy as np
from sklearn.metrics import classification_report

y_val_adjusted = y_val - 1 

y_val_pred = model.predict(X_val)
y_val_pred_classes = np.argmax(y_val_pred, axis=1) 

print("Classification Report:")
print(classification_report(y_val_adjusted, y_val_pred_classes))


# In[22]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

y_pred = np.argmax(model.predict(X_val), axis=-1)

cm = confusion_matrix(y_val_adjusted, y_pred)

class_names = ["colon_aca", "colon_n", "lung_aca", "lung_n", "lung_scc"]

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, 
            yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.show()


# **fine tuning**

# In[23]:


model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))


# In[24]:


optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[25]:


y_train_adjusted = y_train - 1
y_val_adjusted = y_val - 1

history = model.fit(X_train, y_train_adjusted, epochs=15, validation_data=(X_val, y_val_adjusted), batch_size=32)

val_loss, val_accuracy = model.evaluate(X_val, y_val_adjusted)
print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")


# In[26]:


import matplotlib.pyplot as plt

# Extract loss and accuracy from history
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
epochs = range(1, len(train_loss) + 1)

# Plotting Loss vs. Epochs
plt.figure(figsize=(12, 5))

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracy, label='Training Accuracy', color='blue')
plt.plot(epochs, val_accuracy, label='Validation Accuracy', color='orange')
plt.title('Accuracy vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()


# Plot loss
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, label='Training Loss', color='blue')
plt.plot(epochs, val_loss, label='Validation Loss', color='orange')
plt.title('Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()


# In[27]:


import numpy as np
from sklearn.metrics import classification_report

y_val_adjusted = y_val - 1  

y_val_pred = model.predict(X_val)
y_val_pred_classes = np.argmax(y_val_pred, axis=1) 

print("Classification Report:")
print(classification_report(y_val_adjusted, y_val_pred_classes))


# In[28]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

y_pred = np.argmax(model.predict(X_val), axis=-1)

cm = confusion_matrix(y_val_adjusted, y_pred)

class_names = ["colon_aca", "colon_n", "lung_aca", "lung_n", "lung_scc"]

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, 
            yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.show()


# In[ ]:




