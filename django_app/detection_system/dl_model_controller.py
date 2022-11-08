"""
Original author(s): Linus Åberg, Linus Ivarsson
Modified by: Sandra Smoler Eisenberg, Johann Tammen

File purpose: This file contains the logic to control the dl model.
(retrain, train, deploy, data preparation, etc)

"""
# import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import numpy as np
from .database_connector import DB_Connection
import tensorflow as tf
import keras
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.densenet import DenseNet121
from keras.models import Model, load_model
import os
from .variables import *
from .models import DLModel, ModelMetrics, DeployedModel
from django.conf import settings

"""
Method to fire model retraining
Will connect to db and select all data within our dataset table
And call the functions for re-training
Author(s): Linus Ivarsson, Linus Åberg, Sandra Smoler Eisenberg, Johann Tammen
"""
def fire():
    db_conn = DB_Connection('./db.sqlite3')
    cursor = db_conn.cursor()
    # Load the data from DB, need to change the hmnist to our enteire dataset table
    cancer_data = pd.read_sql_query("SELECT * FROM detection_system_datasetimage", db_conn)
    cancer_data.drop(columns="dataset_id", inplace=True)
    if cancer_data.empty:
        return "Error"
    cancer_data.drop(columns="id", inplace=True)
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_dataset(cancer_data)
    model, performance_metrices = retrain_model(X_train, X_val, X_test, y_train, y_val, y_test)

    version = save_dl_model(model, performance_metrices)
    return version


"""
# Prepare our dataset for training
# Will split the entire dataset into trainset, validationset and testset
# As well as withdraw the target label and reshape so that our model can
# process the sets.
Author(s): Linus Åberg
"""
def prepare_dataset(cancer_data):
    # Used to make sure that the split is the same with multiple calls
    random_state_split = 12345

    test_size = 0.1
    validation_size = 0.2

    train_cancer_data, test_cancer_data = train_test_split(cancer_data, test_size=test_size,
                                                           random_state=random_state_split)

    # Split the train_cancer_data set into training and validation set. training = 80%, validation = 20%
    train_cancer_data, val_cancer_data = train_test_split(train_cancer_data, test_size=validation_size,
                                                          random_state=random_state_split)

    # Split the datasets into features and target

    # Split training_cancer set into feature and target set
    train_cancer_features = train_cancer_data.drop('label',
                                                   axis=COLUMNSAXIS)  # drop() creates a copy and does not affect original data
    train_cancer_target = train_cancer_data["label"].copy()
    train_cancer_target.columns = ['label']

    # Split test_cancer set into feature and target set
    test_cancer_features = test_cancer_data.drop('label',
                                                 axis=COLUMNSAXIS)  # drop() creates a copy and does not affect original data
    test_cancer_target = test_cancer_data["label"].copy()
    test_cancer_target.columns = ['label']

    # Split val_cancer set into feature and target se
    val_cancer_features = val_cancer_data.drop('label',
                                               axis=COLUMNSAXIS)  # drop() creates a copy and does not affect original data
    val_cancer_target = val_cancer_data["label"].copy()
    val_cancer_target.columns = ['label']

    # Transform datasets into arrays
    X_training = np.array(train_cancer_features, dtype=DTYPE)
    X_testing = np.array(test_cancer_features, dtype=DTYPE)
    X_validating = np.array(val_cancer_features, dtype=DTYPE)

    y_train = np.array(train_cancer_target, dtype=DTYPE)
    y_test = np.array(test_cancer_target, dtype=DTYPE)
    y_val = np.array(val_cancer_target, dtype=DTYPE)

    # Prepare the datasets, divide by 255 to get value between 0-1
    X_train = X_training[:]/CSVRANGE
    X_test = X_testing[:]/CSVRANGE
    X_val = X_validating[:]/CSVRANGE

    # Reshape the data for the CNN model
    X_train = X_training.reshape(X_training.shape[0], * (IMAGEWIDTH, IMAGEHEIGHT))
    X_test = X_testing.reshape(X_testing.shape[0], * (IMAGEWIDTH, IMAGEHEIGHT))
    X_val = X_validating.reshape(X_validating.shape[0], * (IMAGEWIDTH, IMAGEHEIGHT))

    # Padding the sets to make them 32x32
    X_train = tf.pad(tensor=X_train, paddings=[[0, 0], [PADDING, PADDING], [PADDING, PADDING]])
    X_test = tf.pad(tensor=X_test, paddings=[[0, 0], [PADDING, PADDING], [PADDING, PADDING]])
    X_val = tf.pad(tensor=X_val, paddings=[[0, 0], [PADDING, PADDING], [PADDING, PADDING]])

    # Adding the 3 channel to shape so the DenseNEt121 can accept our data
    X_train = np.repeat(X_train[..., np.newaxis], CHANNELS, CHANNELPOSITION)
    X_val = np.repeat(X_val[..., np.newaxis], CHANNELS, CHANNELPOSITION)
    X_test = np.repeat(X_test[..., np.newaxis], CHANNELS, CHANNELPOSITION)

    return X_train, X_val, X_test, y_train, y_val, y_test

'''
Method to retrain model based on the current active model
it takes in the dataset to be trained on and outputs
the new model, as well as the precision score, recall
score and f1 score.
Author(s): Linus Ivarsson, Linus Åberg
'''
def retrain_model(X_train, X_val, X_test, y_train, y_val, y_test):
    model = load_model(os.path.join('detection_system/model_versions/',
                                    str(DLModel.objects.get(active=True))) + '.h5')
    history = model.fit(X_train, y_train,
                        epochs=EPOCHS,
                        validation_data=(X_val, y_val),
                        batch_size=BATCHSIZE)

    predicted_classes = np.argmax(model.predict(X_test), COLUMNSAXIS)

    cf_matrix = confusion_matrix(y_test, predicted_classes)

    prec = precision_score(y_test, predicted_classes, average='macro')
    rec = recall_score(y_test, predicted_classes, average='macro')
    f1 = f1_score(y_test, predicted_classes, average='macro')

    prec_rec_f1 = {'Precision': [prec], 'Recall': [rec], 'F1': [f1], 'ConfusionMatrix': [cf_matrix]}
    df_prec_rec_f1 = pd.DataFrame(data=prec_rec_f1)
    return model, df_prec_rec_f1

"""
Method to train our CNN.
It uses the DenseNet121 convolutional layer
and then go through our fully connected deep learning model
Author(s): Linus Ivarsson, Linus Åberg
"""

def train_model(X_train, X_val, X_test, y_train, y_val, y_test):
    base_model = DenseNet121(weights='./densenet.hdf5',
                             include_top=False,
                             input_shape=(PADDEDWIDTH, PADDEDHEIGHT, CHANNELS))

    # Connecting all of the model layers
    x = base_model.output  # (None, 1, 1, 1024)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)

    predictions = Dense(OUTPUTNEURONS, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam',
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train,
                        epochs=EPOCHS,
                        validation_data=(X_val, y_val),
                        batch_size=BATCHSIZE)

    predicted_classes = np.argmax(model.predict(X_test), COLUMNSAXIS)

    cf_matrix = confusion_matrix(y_test, predicted_classes)

    prec = precision_score(y_test, predicted_classes, average='macro')
    rec = recall_score(y_test, predicted_classes, average='macro')
    f1 = f1_score(y_test, predicted_classes, average='macro')

    prec_rec_f1 = {'Precision': [prec], 'Recall': [rec], 'F1': [f1], 'ConfusionMatrix': [cf_matrix]}
    df_prec_rec_f1 = pd.DataFrame(data=prec_rec_f1)
    return model, df_prec_rec_f1


"""
Method to save the new dl_model in the system
Will save the model and the performance metrices for the model
Author(s): Linus Ivarsson, Linus Åberg
"""

def save_dl_model(model, performance_metrics):
    # Dynamically name the model versions
    # Get and sort all files in the folder
    files = os.listdir("./detection_system/model_versions")
    files.sort()
    # Incrementing version number
    v_num = 1
    """
    Loop description: Goes through all lists in the folder to compare their names to the new models
    """
    for file in files:
        # If the model name already exists
        if 'v.' + str(v_num) + '.h5' == file:
            # Increment version number
            v_num = v_num + 1

    filename = 'v.' + str(v_num)

    # Save the model as a h5 file
    model.save('./detection_system/model_versions/' + filename + '.h5')

    # Save model version in db
    dl_model = DLModel(version=filename)
    dl_model.save()

    # Store the metrics in database
    metrics = ModelMetrics(precision=performance_metrics['Precision'], recall=performance_metrics['Recall'],
                           f1=performance_metrics['F1'], dl_model=dl_model)
    metrics.save()
    return filename


# Updated the active field in dl_models table to the one that is deployed for making predictions
# Authors: Sandra Smoler Eisenberg, Johann Tammen
def set_active_dl_model(version, roll_back):
    # Set all models in DB to active=False (un-deployed)
    DLModel.objects.all().update(active=False)
    # Set the re-trained model to active (deployed)
    last_model = DLModel.objects.get(version=version)
    last_model.active = True
    last_model.save()

    # Only add an entry in deployed models table if the method was not triggered in a rollback
    if not roll_back:
        deployed_model = DeployedModel(model=last_model)
        deployed_model.save()


# Method to rollback the deployed dl_model to the previous one
# Authors: Sandra Smoler Eisenberg, Johann Tammen
def rollback():
    amount_deployed_models = DeployedModel.objects.count()

    # Only rollback if there is at least 2 deployed model in the system
    if amount_deployed_models > 1:
        previous_deployed_model = DeployedModel.objects.order_by('-id')[1]
        set_active_dl_model(previous_deployed_model.model, True)
        old_model = DeployedModel.objects.order_by('-id')[0]
        old_model.delete()


# Method to delete all DL models in the system except the default model (v.1)
# Authors: Sandra Smoler Eisenberg
def delete_dl_models():
    default_version = 'v.1.h5'
    # Remove all model files except default model
    folder = settings.MODEL_VERSION_ROOT
    for file in os.listdir(folder):
        if file != default_version:
            os.remove(os.path.join(folder, file))

    # Remove all DeployedModels from the db
    DeployedModel.objects.all().delete()
    # Remove all ModelMetrics from the db
    ModelMetrics.objects.all().delete()
    # Remove all DLModels from the db
    DLModel.objects.all().delete()
