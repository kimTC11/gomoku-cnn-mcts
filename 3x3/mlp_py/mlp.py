
# Importing Libraries:
import os
# Disable TensorFlow / Abseil logging
os.environ['ABSL_LOG'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Force TensorFlow to use CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import pandas as pd
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# for preprocessing the data:
from sklearn.preprocessing import LabelEncoder, StandardScaler
from category_encoders.cat_boost import CatBoostEncoder
from sklearn.model_selection import train_test_split

# importing the neural network libraries:
from keras.optimizers import *
from keras.losses import binary_crossentropy
from keras.models import Sequential
from keras.layers import Dense

# importing a classifier from xgboost:
from xgboost import XGBClassifier

# importing metrics to measure our accuracy:
from sklearn.metrics import accuracy_score

start_time = time.time()
logging.info('Reading data...')

# Preprocessing the Data:

data = pd.read_csv("/Users/kimcuong/source/python/tictactoe-cnn-mcts/3x3/data/tic-tac-toe 2.csv")
logging.info('Data shape: %s', data.shape)

logging.info('Checking for NaN values...')
nan_count = data.isnull().sum().sum()
logging.info('Total NaN values: %d', nan_count)

logging.info('Defining labels and dropping target column...')
y = data['class']
data.drop(['class'], inplace=True, axis=1)

logging.info('Encoding labels...')
label = LabelEncoder()
y = label.fit_transform(y)

logging.info('CatBoost encoding features...')
cbe = CatBoostEncoder()
data = cbe.fit_transform(data, y)

logging.info('Splitting train/test sets...')
train, test, ytrain, ytest = train_test_split(data, y, test_size=0.4, train_size=0.6)
logging.info('Train shape: %s, Test shape: %s', train.shape, test.shape)

logging.info('Building neural network model...')
model = Sequential([
    Dense(256, activation='relu', input_shape=(9,)),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(32, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(metrics=['accuracy'], loss='binary_crossentropy', optimizer='Adam')

logging.info('Training neural network...')
fit_start = time.time()
model.fit(train, ytrain, epochs=40, validation_data=(test, ytest))
fit_end = time.time()
logging.info('Neural network training time: %.2f seconds', fit_end - fit_start)

# Concluding with fitting the neural network, I can say that it is good but we can do better since in my opinion a 1000 examples aren't exactly enough to train a plain artificial neural network with and get tinkerable results (pardon me if I am wrong I am new to this too!). The accuracy we got is almost 95% on the training set while the accuracy on the test set is 96% .

# Lets try predicting with XGBoost Classifier now:

logging.info('Training XGBoost classifier...')
xg = XGBClassifier(n_estimators=350)
xg.fit(train, ytrain)
xgPreds = xg.predict(test)
xgb_acc = accuracy_score(xgPreds, ytest)
logging.info('XGBoost test accuracy: %.4f', xgb_acc)

# Model Export and Saving

logging.info('Saving models and encoders...')
import pickle
os.makedirs('models', exist_ok=True)
model.save('models/neural_network_model.h5')
logging.info("Neural Network model saved to: models/neural_network_model.h5")
with open('models/xgboost_model.pkl', 'wb') as f:
    pickle.dump(xg, f)
logging.info("XGBoost model saved to: models/xgboost_model.pkl")
with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(label, f)
logging.info("Label Encoder saved to: models/label_encoder.pkl")
with open('models/catboost_encoder.pkl', 'wb') as f:
    pickle.dump(cbe, f)
logging.info("CatBoost Encoder saved to: models/catboost_encoder.pkl")

# How to Load the Models Later

# How to Load the Models Later
from keras.models import load_model
import pickle
# Load Neural Network
# loaded_nn_model = load_model('models/neural_network_model.h5')
# Load XGBoost model
# with open('models/xgboost_model.pkl', 'rb') as f:
#     loaded_xgb_model = pickle.load(f)
# Load Label Encoder
# with open('models/label_encoder.pkl', 'rb') as f:
#     loaded_label_encoder = pickle.load(f)
# Load CatBoost Encoder
# with open('models/catboost_encoder.pkl', 'rb') as f:
#     loaded_catboost_encoder = pickle.load(f)
logging.info("💡 Use the code above (uncommented) to load models in new projects!")
logging.info("📁 All models are saved in the 'models/' directory")

# Model Evaluation: Confusion Matrix & Metrics

logging.info('Evaluating neural network predictions...')
from sklearn.metrics import confusion_matrix, classification_report
nn_preds = (model.predict(test) > 0.5).astype(int)
cm = confusion_matrix(ytest, nn_preds)
logging.info("Confusion Matrix:\n%s", cm)
cr = classification_report(ytest, nn_preds)
logging.info("Classification Report:\n%s", cr)

# Visualizing Training Curves and Metrics

logging.info('Training again for history and plotting curves...')
import matplotlib.pyplot as plt
history = model.fit(train, ytrain, epochs=40, validation_data=(test, ytest), verbose=0)
logging.info('Plotting loss curve...')
plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
logging.info('Plotting accuracy curve...')
plt.figure(figsize=(8,4))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
logging.info('Model summary:')
model.summary()

end_time = time.time()
logging.info('Total script execution time: %.2f seconds', end_time - start_time)
