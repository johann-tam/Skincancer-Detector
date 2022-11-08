"""
Original author(s): Johann Tammen (Moving methods into files)
Modified by: Sandra Smoler Eisenberg (Add init_dataset() to the file)
Linus Ivarsson

File purpose: Gather methods needed to initialize the system when booting
"""

# import libraries
from .models import DLModel, ModelMetrics, DeployedModel, DataSet
from . import csv_handler


# Initialize default dl_model, generate according entries in the db
# Authors: Sandra Smoler Eisenberg, Johann Tammen
def init_dl_model():
    print("init dl_model ...")
    amount_dl_models = DLModel.objects.count()

    # Check if we have ml_models in the system
    if amount_dl_models < 1:
        # Generate entry in DLmodel for default model version
        dl_model = DLModel(version="v.1", active=True)
        dl_model.save()

        # Set the metrics to 0, so that when retraining the model in the system, the new version is
        # better and updates the the used model. Original values for the current v.1. trained on the 8000 train images:
        # Precision: 0.17924465614683008, Recall:0.17487772750298283 , F1: 0.1659947162734738
        metrics = ModelMetrics(precision=0.17924465614683008, recall=0.17487772750298283, f1=0.1659947162734738, dl_model=dl_model)
        metrics.save()

        # Generate entry in the deployed model table
        deployed_model = DeployedModel(model=dl_model)
        deployed_model.save()
    print("done!")


# Initialize dataset used for training the default dl_model, generate according entries in the db
# Authors: Sandra Smoler Eisenberg
def init_dataset():
    print("init dataset ...")
    # If the dataset has already been initialized we do not want to do it again
    amount_datasets = DataSet.objects.count()
    if amount_datasets < 1:
        # Store the file in a variable and send it to the upload_dataset method
        train_set = open('datasets/train-set.csv', 'r')
        description = "Initial dataset containing 8000 entries."
        csv_handler.upload_dataset(train_set, description)
        # Retrieve the uploaded dataset from the db and give it the proper name
        uploaded_train_set = DataSet.objects.all()[0]
        uploaded_train_set.name = 'train-set.csv'
        uploaded_train_set.save()
    print("done!")
