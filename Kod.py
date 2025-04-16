# Drive bağlantısı
from google.colab import drive
drive.mount('/content/drive')

# Zip dosyasına erişim
import zipfile

zip_path = "/content/drive/MyDrive/colon_image_sets.zip"

extract_path = "colon_dataset"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Gerekli kütüphaneler
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Uyarıları devre dışı bırak
import warnings
warnings.filterwarnings("ignore")
plt.style.use('ggplot')

# Veri setinin yolunu belirle
data_dir = "colon_dataset"
filepaths = []
labels = []

# Klasörler içinde gezerek dosya yollarını oluştur
for sub_folder in os.listdir(data_dir):
    sub_folder_path = os.path.join(data_dir, sub_folder)
    if os.path.isdir(sub_folder_path):
        for file in os.listdir(sub_folder_path):
            if file.endswith((".jpg", ".jpeg", ".png")):
                file_path = os.path.join(sub_folder_path, file)
                filepaths.append(file_path)
                labels.append(sub_folder)

# Verileri DataFrame'e dönüştür
df = pd.DataFrame({"filepaths": filepaths, "labels": labels})
print(df.head())
print(f"Toplam veri sayısı: {len(df)}")

# Veriyi eğitim, doğrulama ve test setlerine ayır
train_df, temp_df = train_test_split(df, train_size=0.8, shuffle=True, random_state=123)
valid_df, test_df = train_test_split(temp_df, train_size=0.5, shuffle=True, random_state=123)
print(f"Train Set: {len(train_df)}, Valid Set: {len(valid_df)}, Test Set: {len(test_df)}")

# Model için görüntü ayarlarını tanımla
batch_size = 32
img_size = (224, 224)
channels = 3
img_shape = (img_size[0], img_size[1], channels)

# Veri artırma işlemi
tr_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Validasyon ve test için veri oluştur
ts_gen = ImageDataGenerator(rescale=1./255)

# Veri seti üzerinden veri akışı oluştur
train_gen = tr_gen.flow_from_dataframe(train_df, x_col='filepaths', y_col='labels',
                                       target_size=img_size, class_mode='categorical',
                                       color_mode='rgb', shuffle=True, batch_size=batch_size)

valid_gen = ts_gen.flow_from_dataframe(valid_df, x_col='filepaths', y_col='labels',
                                       target_size=img_size, class_mode='categorical',
                                       color_mode='rgb', shuffle=True, batch_size=batch_size)

test_gen = ts_gen.flow_from_dataframe(test_df, x_col='filepaths', y_col='labels',
                                      target_size=img_size, class_mode='categorical',
                                      color_mode='rgb', shuffle=False, batch_size=batch_size)

# Sınıf bilgilerini al
g_dict = train_gen.class_indices
classes = list(g_dict.keys())
print(f"Sınıflar: {classes}")

# EfficientNetB3 temel modelini yükle
base_model = tf.keras.applications.EfficientNetB3(include_top=False, weights="imagenet",
                                                  input_shape=img_shape, pooling='none')

# Kendi modelimizi oluşturma işlemi
model = Sequential([
    base_model,
    BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
    GlobalAveragePooling2D(),
    Dense(512, kernel_regularizer=regularizers.l2(0.01), activation='relu'),
    Dropout(rate=0.5),
    Dense(len(classes), activation='softmax')
])

# Modeli derle
model.compile(tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy', metrics=['accuracy'])

# Modeli özetle
model.summary()

# Erken durdurma ve öğrenme oranı azaltma için callback'ler
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=0.00001)

# Modeli eğit
epochs = 10
history = model.fit(train_gen, epochs=epochs, verbose=1, validation_data=valid_gen, shuffle=False,
                    callbacks=[early_stopping, reduce_lr])

# Modeli kaydet
tf.saved_model.save(model, "/content/drive/MyDrive/colon_cancer_model")
print("Model kaydedildi: colon_cancer_model")

# Eğitim sürecini görselleştir
plt.figure(figsize=(12, 5))

# Doğruluk grafiği
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.title('Model Doğruluk Oranları')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()

# Kayıp grafiği
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title('Model Kayıp Değerleri')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()

plt.show()

# En iyi modeli seç
best_model = model

# Test seti üzerinde değerlendirme yap
test_loss, test_accuracy = best_model.evaluate(test_gen)
print(f"Test Kaybı: {test_loss:.4f}")
print(f"Test Doğruluğu: {test_accuracy:.4f}")

# Test seti üzerinde tahminler yap
test_images, test_labels = next(test_gen)
predictions = best_model.predict(test_images)
predictions = np.argmax(predictions, axis=1)

# Tahmin edilen ve gerçek etiketleri oluştur
predicted_labels = [classes[k] for k in predictions]
true_labels = [classes[np.argmax(label)] for label in test_labels]

# Test görüntülerini ve tahminleri görselleştir
num_images = min(32, len(test_images))
fig, axes = plt.subplots(nrows=4, ncols=8, figsize=(20, 20))

for i, ax in enumerate(axes.flat):
    if i < num_images:
        img = (test_images[i] * 255).astype(np.uint8)
        ax.imshow(img)
        true_label = true_labels[i]
        pred_label = predicted_labels[i]
        color = "green" if true_label == pred_label else "red"
        ax.set_title(f"Gerçek: {true_label}\nTahmin: {pred_label}", color=color)
        ax.axis('off')

# Tüm test seti üzerinde tahmin yap
all_predictions = best_model.predict(test_gen)
y_pred = np.argmax(all_predictions, axis=1)
y_true = test_gen.classes
class_names = list(test_gen.class_indices.keys())

# Grafikleri ayarla ve göster
plt.tight_layout()
plt.show()
