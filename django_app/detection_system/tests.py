"""
Original author(s): Linus Åberg, Sandra Smoler Eisenberg
Modified by: Linus Ivarsson
File purpose: Unit tests for the system
"""
# Imports
from django.test import TestCase
from .models import UserImage, DataSet, DLModel
from django.test import Client
from . import csv_handler
from . import system_initiator, dl_model_evaluator
import pandas as pd
from pandas._testing import assert_frame_equal


# Test that an uploaded image is stored as expected
# Authors: Linus Åberg, Sandra Smoler Eisenberg
class ImageTest(TestCase):
    def create_image(self, title="TestImage.jpg", image="../media/images/TestImage.jpg", prediction=2):
        return UserImage.objects.create(title=title, image=image, prediction=prediction)

    def test_checkData(self):
        test_image = self.create_image()
        self.assertEqual(test_image.title, "TestImage.jpg")
        self.assertEqual(test_image.image, "../media/images/TestImage.jpg")
        self.assertEqual(test_image.prediction, 2)


# Test that web client responds with code 200
# Authors: Linus Åberg, Sandra Smoler Eisenberg
class ClientTest(TestCase):
    def setUp(self):
        self.client = Client()

    def test_details(self):
        response = self.client.get('')
        self.assertEqual(response.status_code, 200)


# Test to make sure we can make predictions
# Checks if response code == 200
class PostImageTest(TestCase):
    def setUp(self):
        self.client = Client()

    def create_image(self, image="../media/images/TestImage.jpg"):
        return UserImage.objects.create(image=image)

    def test_post_picture(self):
        test_image = self.create_image()
        response = self.client.post('', {'image': test_image.image})
        self.assertEqual(response.status_code, 200)


'''
# Test to upload datasets
# Checks if response code == 200
class SetUploadTest(TestCase):
    def setUp(self):
        self.client = Client()

    def test_post_dataset(self):
        myFile = open("datasets/set-1.csv", 'r')
        response = self.client.post('/datasets/', {'file': myFile})
        print(response)
        self.assertEqual(response.status_code, 200)
'''
"""
# Test to retrain model
# Checks if response code == 200
class ReTrainTest(TestCase):
    def setUp(self):
        self.client = Client()

    def test_retrain(self):
        response = self.client.post('/models/', {'action': 'retrain'})
        self.assertEqual(response.status_code, 200)
"""

'''''
# Test to rollback to previous model.
# Checks if response code == 200
class RollbackTest(TestCase):
    def setUp(self):
        self.client = Client()

    def test_rollback(self):
        response = self.client.post('/models/', {'action': 'rollback'})
        self.assertEqual(response.status_code, 200)
'''

"""
Test to try the upload dataset function in csv-handler
This test uses a valid dataset and should therefore not get "Fail" or be a non object returned
Author: Linus Ivarsson
"""


class UploadDatasetSuccessTest(TestCase):
    def test_upload_dataset(self):
        description = "Test description"
        affected_rows, test_dataset_id = csv_handler.upload_dataset("./datasets/set-1.csv", description)
        self.assertIsNot(test_dataset_id, 'Fail')
        self.assertIsNotNone(test_dataset_id)


"""
Test to try the upload dataset function in csv-handler
This test uses a invalid dataset and should therefore not get "Fail" returned
Author: Linus Ivarsson
"""


class UploadDatasetFailTest(TestCase):
    def test_upload_dataset(self):
        description = "Test description"
        test_dataset_id = csv_handler.upload_dataset("./datasets/testing-datasets/784_columns.csv", description)
        expected = 'Invalid dataset'
        self.assertEqual(test_dataset_id, expected)


"""
Test to try the validate_dataset function in csv_handler
This test uses a valid dataset and should therefore get a True (1) response
Author: Linus Ivarsson
"""


class ValidateDatasetSuccessTest(TestCase):
    def test_validate_dataset(self):
        # Preprocessing of data to have the expected schema
        df = pd.read_csv("./datasets/set-1.csv")
        df.columns = df.columns.str.strip()
        affected_rows, data = csv_handler.validate_dataset(df)
        assert_frame_equal(df, data)


"""
Test to try the validate_dataset function in csv_handler
This test uses a invalid dataset (784 columns) and should therefore get a False (0) response
Author: Linus Ivarsson
"""


class ValidateDatasetFailColumnTest(TestCase):
    def test_validate_dataset(self):
        # Preprocessing of data to have the expected schema
        df = pd.read_csv("./datasets/testing-datasets/784_columns.csv")
        df.columns = df.columns.str.strip()
        self.assertFalse(csv_handler.validate_dataset(df))


"""
Test to try the validate_dataset function in csv_handler
This test uses a invalid dataset (no label column) and should therefore get a False (0) response
Author: Linus Ivarsson
"""


class ValidateDatasetFailLabelTest(TestCase):
    def test_validate_dataset(self):
        # Preprocessing of data to have the expected schema
        df = pd.read_csv("./datasets/testing-datasets/no_label_column.csv")
        df.columns = df.columns.str.strip()
        self.assertIsNone(csv_handler.validate_dataset(df))


"""
Test to try the initialization of a DL-model
The test checks if the amount of models is not 0 after the function call
Author: Linus Ivarsson
"""


class InitDLModelSuccessTest(TestCase):
    def test_init_dl_model(self):
        system_initiator.init_dl_model()
        amount_dl_models = DLModel.objects.count()
        self.assertIsNot(amount_dl_models, 0)


"""
Test to try the initialization of a dataset
The test checks if the amount of datasets is not 0 after the function call
Author: Linus Ivarsson
"""


class InitDatasetSuccessTest(TestCase):
    def test_init_dataset(self):
        system_initiator.init_dataset()
        amount_datasets = DataSet.objects.count()
        self.assertIsNot(amount_datasets, 0)


"""
Test to try the evaluation method on new data
The test checks the model performance mertics of the model on the new data is calculated and therefore not 0
Author: Linus Ivarsson
"""


class EvaluateModelNewDataSuccessTest(TestCase):
    def test_evaluate_model(self):
        description = "Test description"
        # Upload new dataset and store the dataset_id
        affected_rows, test_dataset_id = csv_handler.upload_dataset("./datasets/set-1.csv", description)
        # Initialize a model
        system_initiator.init_dl_model()
        # Store metrics
        prec, rec, f1 = dl_model_evaluator.evaluate_model_on_new_data(test_dataset_id)
        # Make assertions
        self.assertIsNot(prec, 0)
        self.assertIsNot(rec, 0)
        self.assertIsNot(f1, 0)


"""
Test to try the evaluation functions robustness when receiving an error message
The test checks the returned metrics are all 0's when receiving an error message.
Author: Linus Ivarsson
"""


class EvaluateModelNewDataFailTest(TestCase):
    def test_evaluate_model(self):
        error_message = 0
        description = "Test description"
        # Upload new invalid dataset and store the dataset_id
        test_dataset_id = csv_handler.upload_dataset("./datasets/testing-datasets/no_label_column.csv",
                                                                    description)
        # Initialize a model
        system_initiator.init_dl_model()
        # Store metrics
        prec, rec, f1 = dl_model_evaluator.evaluate_model_on_new_data(test_dataset_id)
        # Make assertions
        self.assertEqual(prec, error_message)
        self.assertEqual(rec, error_message)
        self.assertEqual(f1, error_message)
