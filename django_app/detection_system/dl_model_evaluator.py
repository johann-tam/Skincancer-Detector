"""
Original author(s): Johann Tammen (Moving methods into files)
Modified by:

File purpose: Gather methods used for calculating and comparing the performances of dl_models
"""

# import libraries
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import numpy as np
import tensorflow as tf
import keras
import os
from .variables import *
from .models import DLModel, ModelMetrics
from .database_connector import DB_Connection


# Function to compare whether the new re-trained is performing better than the currently deployed model
# return 1 when the new version is better than the currently deployed one, return 0 when the deployed model is better
# than the new one
# Authors: Sandra Smoler Eisenberg, Johann Tammen
def compare_dl_models(version):
    active_model = DLModel.objects.get(active=True)
    latest_model = DLModel.objects.get(version=version)

    # Get the metrics of each model by the ModelMetrics foreign key
    active_model_metrics = ModelMetrics.objects.get(dl_model_id=active_model.pk)
    latest_model_metrics = ModelMetrics.objects.get(dl_model_id=latest_model.pk)

    # return true if the performance (f1-score) of the new model is better than the previously deployed one,
    # else return false
    return active_model_metrics.f1 < latest_model_metrics.f1

""" 
Method description: Calulates precision, recall and F1-score on the newly uploaded dataset with the currently active 
model. Returns precision, recall and f1-score.
Parameter description: Input: id of the dataset to compare against the currently deployed model. 
Author: Linus Ivarsson, Linus Åberg
Modified by: Johann Tammen
"""
def evaluate_model_on_new_data(dataset_id):

    error_code = 0

    if dataset_id != 'Invalid dataset':
        db_conn = DB_Connection('./db.sqlite3')

        # Retrieve all entries of the new dataset
        new_dataset = pd.read_sql_query("SELECT * FROM detection_system_datasetimage WHERE dataset_id = "
                                        + str(dataset_id), db_conn)
        # Drop unnecessary columns
        new_dataset.drop(columns="dataset_id", inplace=True)
        if new_dataset.get("id") is not None:
            new_dataset.drop(columns="id", inplace=True)
        X_features = new_dataset.drop('label', axis=1)  # drop() creates a copy and does not affect original data
        # Store targets
        y_target = new_dataset["label"].copy()

        # Prepare data for model
        X_validating = np.array(X_features, dtype=DTYPE)
        y_val = np.array(y_target, dtype=DTYPE)
        X_val = X_validating.reshape(X_validating.shape[0], *(IMAGEWIDTH, IMAGEHEIGHT))
        X_val = tf.pad(tensor=X_val, paddings=[[0, 0], [PADDING, PADDING], [PADDING, PADDING]])
        X_val = np.repeat(X_val[..., np.newaxis], CHANNELS, CHANNELPOSITION)

        # Load currently active model
        model = keras.models.load_model(
            os.path.join('detection_system/model_versions/', str(DLModel.objects.get(active=True))) + '.h5')

        # Store predictions made by model
        predicted_classes = np.argmax(model.predict(X_val), COLUMNSAXIS)

        # Store all performance metrics
        cf_matrix = confusion_matrix(y_val, predicted_classes)
        prec = precision_score(y_val, predicted_classes, average='macro')
        rec = recall_score(y_val, predicted_classes, average='macro')
        f1 = f1_score(y_val, predicted_classes, average='macro')

        return prec, rec, f1

    else:
        # return 0 for metrics when the dataset is not a valid id
        return error_code, error_code, error_code
