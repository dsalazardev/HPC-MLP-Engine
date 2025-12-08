from data_loader import MNISTLoader
import matplotlib.pyplot as plt

path = "/Dataset/archive"
loader = MNISTLoader(path)

print("Cargando MNIST...")
(X_train, y_train), (X_test, y_test) = loader.load_data()

print(f"Forma X_train: {X_train.shape}")
print(f"Forma y_train: {y_train.shape}")

first_image = X_train[:, 0].reshape(28, 28)
label = y_train[0]

plt.imshow(first_image, cmap='gray')
plt.title(f"Etiqueta Real: {label}")
plt.show()