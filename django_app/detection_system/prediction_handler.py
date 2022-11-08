"""
Original author(s): Linus Ivarsson
Modified by: Sandra Smoler Eisenberg, Linus Åberg

File purpose: Handles the list of predictions made by the model
"""

import numpy as np
from . import variables

""" 
Method description: Takes in a prediction array, transform values to percentages and returns the top three 
label names and the percentages. 
Parameter description: List of predictions values
Author(s): Linus Ivarsson, Sandra Smoler Eisenberg, Linus Åberg
"""


def extract_top_predictions(prediction):

    # Transform values into percentages and store in separate list
    prediction_list = prediction[0] * 100
    # Make copy of prediction list since it will be altered
    prediction_list_copy = prediction[0]
    # For all three labels - extract label for the highest current percentage then remove from list
    high_label = np.argmax(prediction_list_copy)
    prediction_list_copy[high_label] = 0
    med_label = np.argmax(prediction_list_copy)
    prediction_list_copy[med_label] = 0
    low_label = np.argmax(prediction_list_copy)
    # Sort prediction_list in descending order and store top 3 predictions
    top_3_predictions = sorted(prediction_list, reverse=True)[:3]
    # Round top 3 predictions to two decimals
    rounded_prediction_values = [round(prediction, 2) for prediction in top_3_predictions]
    # Set labels to the respective name of the type of skin cancer
    label_list = [variables.LABEL_DIRECTORY[high_label], variables.LABEL_DIRECTORY[med_label],
                  variables.LABEL_DIRECTORY[low_label]]
    # set info as the return from prediction_information and send in labels
    info = prediction_information(high_label, med_label, low_label)
    # Zip the two lists so that they can be iterated over simultaneously in the HTML
    prediction = zip(label_list, rounded_prediction_values, info)
    # Return the zipped lists as the prediction
    return prediction


""" 
Method description: Takes in thre labels which corresponds to type of skin abnormality.
It will return a list containing lnks to further information about the abnormality
Parameter description: Top prediction labels
Author(s): Linus Åberg
"""


def prediction_information(high_label, med_label, low_label):
    # Set pred_info to the respective link for additional information
    pred_info = [variables.LINK_DIRECTORY[high_label], variables.LINK_DIRECTORY[med_label],
                 variables.LINK_DIRECTORY[low_label]]

    # list of links
    return pred_info

