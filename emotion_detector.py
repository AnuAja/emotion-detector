import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.datasets import mnist
from sklearn.model_selection import train_test_split


import os

# Load and preprocess the dataset
def load_dataset():
    directory = 'F:/Kuliah/Anugrah/AI/Emotion Detection/train'
    image_paths = []
    labels = []

    label_mapping = {
        'angry': 0,
        'happy': 1,
        'sad': 2,
        'neutral': 3,
    }

    for expression_dir in os.listdir(directory):
        expression_path = os.path.join(directory, expression_dir)
        if os.path.isdir(expression_path):
            for filename in os.listdir(expression_path):
                file_path = os.path.join(expression_path, filename)
                if os.path.isfile(file_path) and filename.endswith((".jpg", ".jpeg", ".png")):
                    image_paths.append(file_path)
                    labels.append(label_mapping[expression_dir])
    X_train, X_test, y_train, y_test = train_test_split(image_paths, labels, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


# Preprocess and reshape the dataset
def preprocess_dataset(X_train, X_test):
    X_train_processed = []
    X_test_processed = []

    for image_path in X_train:
        image = cv2.imread(image_path)
        image = cv2.resize(image, (48, 48))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image / 255.0
        X_train_processed.append(image)

    for image_path in X_test:
        image = cv2.imread(image_path)
        image = cv2.resize(image, (48, 48))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image / 255.0
        X_test_processed.append(image)

    X_train_processed = np.array(X_train_processed)
    X_test_processed = np.array(X_test_processed)

    X_train_processed = np.reshape(X_train_processed, (X_train_processed.shape[0], 48, 48, 1))
    X_test_processed = np.reshape(X_test_processed, (X_test_processed.shape[0], 48, 48, 1))

    return X_train_processed, X_test_processed


# Build the emotion detection model
def build_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# Train the emotion detection model
def train_model(model, X_train, X_test, y_train, y_test):
    y_train = to_categorical(y_train, 4)
    y_test = to_categorical(y_test, 4)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)


# Detect emotions in images
def detect_emotions(model, image):
    image = cv2.resize(image, (48, 48))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image / 255.0
    image = np.reshape(image, (1, 48, 48, 1))

    predictions = model.predict(image)
    emotion_labels = ['Angry', 'Happy', 'Sad', 'Neutral']
    predicted_label = emotion_labels[np.argmax(predictions)]
    confidence = np.max(predictions)

    return predicted_label, confidence


# Main function
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_dataset()
    X_train, X_test = preprocess_dataset(X_train, X_test)
    model = build_model()
    train_model(model, X_train, X_test, y_train, y_test)

    cap = cv2.VideoCapture(1)  

    while True:
        ret, frame = cap.read()  
        predicted_label, confidence = detect_emotions(model, frame)

        cv2.putText(frame, f"Emotion: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the windows
    cap.release()
    cv2.destroyAllWindows()
