import os
from skimage import io
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        label = 0 if filename.startswith('class1') else 1 
        img = io.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img.flatten())  # Aplanar la imagen para usarla directamente como características
            labels.append(label)
    return images, labels

# Cargar imágenes y etiquetas
folder_path = r'C:/Users/jimen/Desktop/Tyre dataset/Tyre dataset/DatosModelo'  # Ruta a tu carpeta de imágenes
images, labels = load_images_from_folder(folder_path)

# Dividir datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Crear y entrenar el modelo SVM
model = make_pipeline(StandardScaler(), SVC(kernel='linear'))
model.fit(X_train, y_train)

# Evaluar el modelo
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)