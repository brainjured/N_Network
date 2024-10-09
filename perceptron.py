import os
import numpy as np
import pickle
from PIL import Image, ImageTk, ImageDraw
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox, StringVar
from tkinter import ttk
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score

class MultilayerPerceptron:
    def __init__(self, input_size, hidden_sizes, num_classes, l_rate=0.001):
        self.whgt = []
        self.biases = []
        self.l_rate = l_rate
        self.whgt.append(np.random.randn(input_size, hidden_sizes[0]) * np.sqrt(2. / input_size))
        self.biases.append(np.zeros((1, hidden_sizes[0])))
        for i in range(1, len(hidden_sizes)):
            self.whgt.append(
                np.random.randn(hidden_sizes[i - 1], hidden_sizes[i]) * np.sqrt(2. / hidden_sizes[i - 1]))
            self.biases.append(np.zeros((1, hidden_sizes[i])))
        self.whgt.append(np.random.randn(hidden_sizes[-1], num_classes) * np.sqrt(2. / hidden_sizes[-1]))
        self.biases.append(np.zeros((1, num_classes)))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def cross_entropy_loss(self, y_true, y_pred):
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-9), axis=1))

    def forward(self, X):
        self.activations = []
        self.z_values = []
        A = X

        for i in range(len(self.whgt) - 1):
            Z = np.dot(A, self.whgt[i]) + self.biases[i]
            A = self.relu(Z)
            self.z_values.append(Z)
            self.activations.append(A)

        Z = np.dot(A, self.whgt[-1]) + self.biases[-1]
        A = self.softmax(Z)
        self.z_values.append(Z)
        self.activations.append(A)
        return A

    def backward(self, X, y, output):
        m = X.shape[0]

        dZ = output - y
        dW = np.dot(self.activations[-2].T, dZ) / m
        db = np.sum(dZ, axis=0, keepdims=True) / m
        self.whgt[-1] -= self.l_rate * dW
        self.biases[-1] -= self.l_rate * db

        for i in reversed(range(len(self.whgt) - 1)):
            dZ = np.dot(dZ, self.whgt[i + 1].T) * self.relu_derivative(self.z_values[i])
            if i == 0:
                dW = np.dot(X.T, dZ) / m
            else:
                dW = np.dot(self.activations[i - 1].T, dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m
            self.whgt[i] -= self.l_rate * dW
            self.biases[i] -= self.l_rate * db

    def train(self, X_train, y_train, X_val, y_val, epochs=700):
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        train_precisions = []
        val_precisions = []
        train_recalls = []
        val_recalls = []

        for epoch in range(epochs):
            output = self.forward(X_train)
            loss = self.cross_entropy_loss(y_train, output)
            train_losses.append(loss)
            self.backward(X_train, y_train, output)

            val_output = self.forward(X_val)
            val_loss = self.cross_entropy_loss(y_val, val_output)
            val_losses.append(val_loss)

            train_accuracy = np.mean(np.argmax(y_train, axis=1) == np.argmax(output, axis=1))
            val_accuracy = np.mean(np.argmax(y_val, axis=1) == np.argmax(val_output, axis=1))
            train_precision = precision_score(np.argmax(y_train, axis=1), np.argmax(output, axis=1), average='weighted', zero_division=0)
            val_precision = precision_score(np.argmax(y_val, axis=1), np.argmax(val_output, axis=1), average='weighted', zero_division=0)
            train_recall = recall_score(np.argmax(y_train, axis=1), np.argmax(output, axis=1), average='weighted', zero_division=0)
            val_recall = recall_score(np.argmax(y_val, axis=1), np.argmax(val_output, axis=1), average='weighted', zero_division=0)

            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)
            train_precisions.append(train_precision)
            val_precisions.append(val_precision)
            train_recalls.append(train_recall)
            val_recalls.append(val_recall)

            if epoch % 10 == 0:
                print(f"Эпоха {epoch + 1}/{epochs}, Потери: {val_loss:.4f}, "
                      f"Точность: {val_accuracy:.4f}, "
                      f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")

        self.plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies,
                          train_precisions, val_precisions, train_recalls, val_recalls)

        return train_losses, val_losses

    def plot_metrics(self, train_losses, val_losses, train_accuracies, val_accuracies,
                     train_precisions, val_precisions, train_recalls, val_recalls):
        epochs = range(1, len(train_losses) + 1)

        plt.figure(figsize=(14, 10))
        plt.subplot(2, 2, 1)
        plt.plot(epochs, val_losses, label='Validation Loss')
        plt.title("График потерь")
        plt.xlabel("Эпоха")
        plt.ylabel("Потери")
        plt.legend()
        plt.subplot(2, 2, 2)
        plt.plot(epochs, val_accuracies, label='Validation Accuracy')
        plt.title("График точности")
        plt.xlabel("Эпоха")
        plt.ylabel("Точность")
        plt.legend()
        plt.subplot(2, 2, 3)
        plt.plot(epochs, val_precisions, label='Validation Precision')
        plt.title("График Precision")
        plt.xlabel("Эпоха")
        plt.ylabel("Precision")
        plt.legend()
        plt.subplot(2, 2, 4)
        plt.plot(epochs, val_recalls, label='Validation Recall')
        plt.title("График Recall")
        plt.xlabel("Эпоха")
        plt.ylabel("Recall")
        plt.legend()

        plt.tight_layout()
        plt.show()

    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)

    def save_whgt(self, filename):
        whgt = {
            'whgt': self.whgt,
            'biases': self.biases
        }
        with open(filename, 'wb') as f:
            pickle.dump(whgt, f)
        print(f"Веса сохранены в файл {filename}")

    def load_whgt(self):
        load_path = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl")])
        if load_path:
            with open(load_path, 'rb') as f:
                data = pickle.load(f)

            if not self.class_labels:
                num_classes = len(data['biases'][-1][0])
                self.class_labels = [f"Class {i}" for i in range(num_classes)]

            if self.perceptron is None:
                input_size = 64 * 64
                hidden_sizes = [128, 64]
                num_classes = len(self.class_labels)
                self.perceptron = MultilayerPerceptron(input_size=input_size, hidden_sizes=hidden_sizes, num_classes=num_classes)
            self.perceptron.whgt = data['whgt']
            self.perceptron.biases = data['biases']
            print(f"Веса загружены из файла {load_path}")
        else:
            print("Не удалось загрузить веса.")

def load_dataset_from_folder(folder_path, img_size=(64, 64)):
    X = []
    y = []
    class_labels = os.listdir(folder_path)
    class_labels.sort()

    for label_idx, label in enumerate(class_labels):
        class_folder = os.path.join(folder_path, label)
        for filename in os.listdir(class_folder):
            file_path = os.path.join(class_folder, filename)
            img = Image.open(file_path).convert('L').resize(img_size)
            img_array = np.array(img).reshape(-1) / 255.0
            X.append(img_array)
            y.append(label_idx)

    X = np.array(X)
    y = np.array(y)

    num_classes = len(class_labels)
    y_one_hot = np.zeros((len(y), num_classes))
    y_one_hot[np.arange(len(y)), y] = 1

    return X, y_one_hot, class_labels

class PerceptronGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Многослойный персептрон GUI")
        self.perceptron = None
        self.dataset_folder = None
        self.class_labels = []
        self.epochs = StringVar(value="700")
        self.l_rate = StringVar(value="0.001")
        self.create_widgets()

    def create_widgets(self):
        hyperparams_frame = ttk.LabelFrame(self.root, text="Гиперпараметры")
        hyperparams_frame.pack(padx=10, pady=10, fill="x")

        ttk.Label(hyperparams_frame, text="Количество эпох:").grid(row=0, column=0, padx=5, pady=5)
        epochs_entry = ttk.Entry(hyperparams_frame, textvariable=self.epochs)
        epochs_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(hyperparams_frame, text="Скорость обучения:").grid(row=1, column=0, padx=5, pady=5)
        l_rate_entry = ttk.Entry(hyperparams_frame, textvariable=self.l_rate)
        l_rate_entry.grid(row=1, column=1, padx=5, pady=5)

        self.load_folder_button = ttk.Button(self.root, text="Выбрать папку с датасетом", command=self.load_folder)
        self.load_folder_button.pack(padx=10, pady=5)

        self.train_button = ttk.Button(self.root, text="Обучить сеть", command=self.train_network)
        self.train_button.pack(padx=10, pady=5)

        self.save_whgt_button = ttk.Button(self.root, text="Сохранить веса", command=self.save_whgt)
        self.save_whgt_button.pack(padx=10, pady=5)

        self.load_whgt_button = ttk.Button(self.root, text="Загрузить веса", command=self.load_whgt)
        self.load_whgt_button.pack(padx=10, pady=5)

        self.open_image_button = ttk.Button(self.root, text="Открыть изображение", command=self.open_image)
        self.open_image_button.pack(padx=10, pady=5)

        self.recognize_button = ttk.Button(self.root, text="Распознать", command=self.recognize_image)
        self.recognize_button.pack(padx=10, pady=5)

        self.metrics_label = ttk.Label(self.root, text="Результат распознавания: Ничего нет", foreground="black")
        self.metrics_label.pack(padx=10, pady=5)

        self.drawing_canvas = tk.Canvas(self.root, width=200, height=200, bg="white")
        self.drawing_canvas.pack(padx=10, pady=10)
        self.drawing_canvas.bind("<B1-Motion>", self.draw)

        self.clear_button = ttk.Button(self.root, text="Очистить рисунок", command=self.clear_canvas)
        self.clear_button.pack(padx=10, pady=5)

        self.image = None
        self.drawing_image = Image.new("L", (64, 64), "white")
        self.drawing = ImageDraw.Draw(self.drawing_image)

    def draw(self, event):
        x, y = event.x, event.y
        self.drawing_canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill="black")
        self.drawing.ellipse([x - 2, y - 2, x + 2, y + 2], fill="black")
        self.metrics_label.config(text="Результат распознавания: Ничего нет")
        self.image = None

    def clear_canvas(self):
        self.drawing_canvas.delete("all")
        self.drawing_image = Image.new("L", (64, 64), "white")
        self.drawing = ImageDraw.Draw(self.drawing_image)
        self.metrics_label.config(text="Результат распознавания: Ничего нет")

    def load_folder(self):
        self.dataset_folder = filedialog.askdirectory()
        print(f"Папка с датасетом выбрана: {self.dataset_folder}")

    def train_network(self):
        if self.dataset_folder:
            X, y, self.class_labels = load_dataset_from_folder(self.dataset_folder, img_size=(64, 64))

            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

            input_size = X.shape[1]
            hidden_sizes = [128, 64]
            num_classes = len(self.class_labels)

            self.perceptron = MultilayerPerceptron(input_size=input_size, hidden_sizes=hidden_sizes,
                                                   num_classes=num_classes)
            train_losses, val_losses = self.perceptron.train(X_train, y_train, X_val, y_val, epochs=int(self.epochs.get()))
        else:
            print("Папка с датасетом не выбрана!")

    def save_whgt(self):
        if self.perceptron:
            save_path = filedialog.asksaveasfilename(defaultextension=".pkl", filetypes=[("Pickle files", "*.pkl")])
            if save_path:
                self.perceptron.save_whgt(save_path)
        else:
            print("Сеть не обучена!")

    def load_whgt(self):
        load_path = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl")])
        if load_path:
            with open(load_path, 'rb') as f:
                data = pickle.load(f)

            if not self.class_labels:
                num_classes = len(data['biases'][-1][0])
                self.class_labels = [f"Class {i}" for i in range(num_classes)]

            if self.perceptron is None:
                input_size = 64 * 64  
                hidden_sizes = [128, 64] 
                num_classes = len(self.class_labels)
                self.perceptron = MultilayerPerceptron(input_size=input_size, hidden_sizes=hidden_sizes, num_classes=num_classes)
            self.perceptron.whgt = data['whgt']
            self.perceptron.biases = data['biases']
            print(f"Веса загружены из файла {load_path}")
        else:
            print("Не удалось загрузить веса.")

    def open_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = Image.open(file_path).resize((64, 64))
            self.show_image_in_canvas(self.image)

    def show_image_in_canvas(self, image):
        self.drawing_canvas.delete("all")
        image_tk = ImageTk.PhotoImage(image)
        self.drawing_canvas.create_image(0, 0, anchor="nw", image=image_tk)
        self.drawing_canvas.image = image_tk
        self.metrics_label.config(text="Результат распознавания: Ничего нет")

    def recognize_image(self):
        if np.count_nonzero(np.array(self.drawing_image) == 255) == (64 * 64) and self.image is None:
            self.metrics_label.config(
                text="Пожалуйста, нарисуйте что-то или откройте изображение перед распознаванием.")
            return

        if self.image:
            img_array = np.array(self.image.convert("L").resize((64, 64))).reshape(1, -1) / 255.0
        else:
            img_array = np.array(self.drawing_image.convert("L").resize((64, 64))).reshape(1, -1) / 255.0

        if self.perceptron:
            prediction = self.perceptron.predict(img_array)
            if prediction[0] < len(self.class_labels):
                predicted_class = self.class_labels[prediction[0]]
                self.metrics_label.config(text=f"Результат распознавания: {predicted_class}")
                print(f"Предсказанное значение для изображения: {predicted_class}")
            else:
                self.metrics_label.config(text="Неизвестный класс.")
        else:
            self.metrics_label.config(text="Сеть не обучена!")

if __name__ == "__main__":
    root = tk.Tk()
    app = PerceptronGUI(root)
    root.mainloop()
