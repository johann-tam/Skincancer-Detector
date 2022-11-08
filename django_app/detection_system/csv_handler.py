"""
Original author(s):  Linus Åberg  <guslinuab@student.gu.se>
Modified by: Johann Tammen <gustammjo@student.gu.se>, Sandra Smoler Eisenberg <smolers@student.chalmers.se>

File purpose: Logic to upload a new dataset to the system
"""

import sqlite3
import pandas as pd
from .models import DataSet, DatasetImage

"""
Returns the id of the uploaded dataset
Modified by: Johann Tammen, Sandra Smoler Eisenberg, Linus Åberg
"""
def upload_dataset(filename, description):
    # Load data
    df = pd.read_csv(filename)
    # strip whitespace from headers
    df.columns = df.columns.str.strip()

    # Check that csv file is well formed
    if validate_dataset(df) is not None:
        affected_rows, valid_dataset = validate_dataset(df)
        # Create new dataset entry in db
        data_set_obj = DataSet(name=filename, description=description)
        data_set_obj.save()

        # Add foreign key dataset_id to the dataframe with the id of the just generated dataset entry
        valid_dataset['dataset_id'] = data_set_obj.pk

        con = sqlite3.connect("db.sqlite3")

        # Drop data into database
        valid_dataset.to_sql("detection_system_datasetimage", con, if_exists='append', index=False)

        con.close()
        return affected_rows, data_set_obj.pk
    else:
        return 'Invalid dataset'


# Perform checks that validates the dataset
# Input: dataset in form of pandas dataframe, Output: original dataset or None if un-valid
# Original Authors: Johann Tammen & Linus Åberg
# Modifier: Sandra Smoler Eisenberg (Add validation for missing/NaN/null, numeric, and in range 0-255)
# Modifier: Linus Åberg (Add functionality to delete "corrupted" rows and still upload)
# + modify return statements
def validate_dataset(dataset):
    amt_columns = 785
    affected_rows = 0
    # Label column is needed in order to be accepted as a new dataset
    if 'label' not in dataset:
        return None

    # The right amount of pixels plus label is needed to be accepted as a new dataset
    if not len(dataset.columns) == amt_columns:
        return None

    # Check that all values in dataset are numeric (covers null/NaN and missing value).
    # Converts dataset to values of True if numeric, else False
    modified_dataset = dataset.apply(lambda x: pd.to_numeric(x, errors='coerce').notnull())
    # Query all rows that are True in the dataset
    for column in dataset:
        modified_dataset.query(column, inplace=True)
    # If rows were removed the dataset contains non-numeric values
    if modified_dataset.shape[0] != dataset.shape[0]:
        return None

    # Check that all values are in range 0-255
    for column in dataset:
        # If not all values are True (between 0-255), there exists invalid values
        if not dataset[column].between(0, 255, inclusive="both").all():
            # Save the index of all rows contain columns that are out of bounds in a list
            ilist = dataset.loc[~dataset[column].between(0, 255, inclusive='both')].index.tolist()
            # Save the number of affected rows in order to show in the Ui
            affected_rows = len(ilist)
            # Drop the rows containing coumns that are out of bounds and save in new dataframe
            new_set = dataset.drop(ilist)
            # Return the new dataframe and affected_rows
            return affected_rows, new_set

    return affected_rows, dataset


# Method to delete all datasets and dataset images
# Authors: Johann Tammen
# Modified by: Sandra Smoler Eisenberg
# Modified to not delete initial dataset since we re-initialize it anyway which takes a lot of time.
def delete_all_datasets():
    # Reference the initial dataset
    initial_dataset = DataSet.objects.all()[0]
    # Delete all DatasetImages that do not belong to initial dataset
    for image in DatasetImage.objects.all():
        if image.dataset != initial_dataset:
            image.delete()
    # Delete all Datasets except the initial dataset
    for dataset in DataSet.objects.all():
        if dataset != initial_dataset:
            dataset.delete()
