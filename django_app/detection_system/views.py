"""
Original author(s): Sandra Smoler Eisenberg
Modified by: Johann Tammen, Linus Ivarsson, Linus Åberg

File purpose: Contains functions which take in a request and returns a response
"""

import os.path
from keras.models import load_model
from django.contrib.admin.views.decorators import staff_member_required
from django.shortcuts import render, redirect
from .forms import *
from .models import DataSet, DLModel, ModelMetrics
from .prediction_handler import *
from . import dl_model_controller, dl_model_evaluator, system_initiator, image_processor, csv_handler, history_controller
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib import messages

# Initialize dl_model in th the db
system_initiator.init_dl_model()
# Initialize dataset used to train the default model (v.1) so that the data exists in the system
system_initiator.init_dataset()


# Function that displays the homepage of the system and
# processes the end-user's uploaded image and returns a prediction result
# Author: Sandra Smoler Eisenberg
# Modified by: Johann Tammen, Linus Ivarsson
def make_prediction(request):
    # Check whether the end-user is logged in or not
    authenticated = authenticate_user(request)
    # Process image uploaded by end-user
    if request.method == 'POST':
        # Import the deployed DL-model to perform the prediction
        dl_model = load_model(os.path.join('detection_system/model_versions/',
                                           str(DLModel.objects.get(active=True))) + '.h5')
        # Store end-user uploaded data in variable
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            img_obj = form.instance
            # Set the name of the image to the name it had when the end-user uploaded it
            img_obj.title = request.FILES['image'].name
            # If user is logged in, store the image to that user
            if authenticated:
                img_obj.user = request.user
            # Once form is saved, the image can be returned and displayed on the UI
            form.save()
            # Processes an Image into a greyscale pixel array size 784
            img_array = image_processor.process_jpg(img_obj.title)
            # Reshapes pixel array into proper format
            processed_img = image_processor.transform_pixel_array(img_array)
            # Makes a prediction for each class for the uploaded image
            prediction = dl_model.predict(processed_img)
            # Add prediction to Image in database
            img_obj.prediction = np.argmax(prediction)
            # Extracts the 3 classes with the highest probability
            top_predictions = extract_top_predictions(prediction)
            # Save userimage to database
            img_obj.save()
            success_message = 'Successful upload'
            # Display the uploaded image and the predictions on the UI
            return render(request, 'home_prediction_bar.html', {'success': success_message,
                                                                'authenticated': authenticated, 'img_obj': img_obj,
                                                                'predictions': top_predictions})
    else:
        # Display the homepage with the upload form on the UI
        form = ImageForm()
    return render(request, 'home.html', {'authenticated': authenticated, 'form': form})


# View to handle csv upload, will take the request body from
# form and pass it to the upload_csv method
# This site is the accesspoint for the admin so it will prompt
# Admin to login hence @staff_member_required.
# Linus Åberg - guslinuab@student.gu.se
@staff_member_required()
def data_management(request):
    # Get all datasets from the DB
    datasets = DataSet.objects.all()
    # Empty string to store potential errormessage
    error_message = ''
    # Declare form as empty csv form
    form = csv_form()
    # Takes a csv file and stores it in the DB
    if request.method == 'POST':
        if request.POST['action'] == "upload":
            # Reassign the form to the the csv file to be uploaded
            form = csv_form(request.POST, request.FILES)
            # Check that there is a file uploaded and that the file is a .csv
            if request.FILES.get('file') is None or not request.FILES.get('file').name.endswith('csv'):
                error_message = "Please select a valid dataset before uploading"
            elif form.is_valid():
                affected_rows, dataset_id = csv_handler.upload_dataset(request.FILES['file'], request.POST.get('DatasetDescription'))
                # Evaluate dataset if it is valid
                if dataset_id != 'Invalid dataset':
                    # Update datasets variable to include newly uploaded one
                    datasets = DataSet.objects.all()
                    # Evaluate active model on the new dataset and store metrics
                    precision, recall, f1score = dl_model_evaluator.evaluate_model_on_new_data(dataset_id)
                    # Check if affected rows are in the value of 1 or higher in order to get the right grammar
                    # If affected rows are 0, then we return render without the message.
                    if affected_rows > 1:
                        message = str(affected_rows) + ' rows contained faulty data and where deleted'
                    elif affected_rows == 1:
                        message = str(affected_rows) + ' row contained faulty data and where deleted'
                    elif affected_rows == 0:
                        return render(request, 'dataset_list.html', {'datasets': datasets, 'precision': precision,
                                                                     'recall': recall, 'f1': f1score})

                    return render(request, 'dataset_list.html', {'datasets': datasets, 'precision': precision,
                                                                 'recall': recall, 'f1': f1score, 'message': message})
                else:
                    error_message = 'Invalid dataset - Must contain 785 columns, label, and numeric values between 0-255'
                    return render(request, 'dataset_list.html', {'datasets': datasets, 'error_message': error_message})
            else:
                form = csv_form()
                return render(request, 'dataset_list.html',
                              {'form': form, 'datasets': datasets, 'error_message': error_message})

        elif request.POST['action'] == "delete":
            # Delete all uploaded datasets and dataset images
            csv_handler.delete_all_datasets()

    return render(request, 'dataset_list.html', {'form': form, 'datasets': datasets, 'error_message': error_message})


# Displays the Models page of the Admin-user.
# Displays the performance of each model
# Allows the Admin-user to Re-train and Rollback the model
# This site is the accesspoint for the admin so it will prompt
# Admin to login hence @staff_member_required.
# Author: Linus Ivarsson, Linus Åberg
# Modified by: Sandra Smoler Eisenberg, Johann Tammen

@staff_member_required()
def model_management(request):
    model_metrics = ModelMetrics.objects.all()
    error_message = ''
    if request.method == 'POST':
        # If the button "Re-train" is pushed - attempt to re-train the model
        if request.POST['action'] == "retrain":
            version = dl_model_controller.fire()
            if version == "Error":
                error_message = "Database empty, please upload a dataset first"
            # Compare the re-trained model to the previous. Deploy the re-trained model if it is better-performing
            elif dl_model_evaluator.compare_dl_models(version):
                dl_model_controller.set_active_dl_model(version, False)
            else:
                print("The re-trained model performed worse than the already existing model. Did not deploy!")
        # If the button "Rollback" is pressed - Rollback to previous model
        elif request.POST['action'] == "rollback":
            dl_model_controller.rollback()
        elif request.POST['action'] == "delete":
            dl_model_controller.delete_dl_models()
            system_initiator.init_dl_model()
    active_model = DLModel.objects.get(active=True)
    return render(request, 'model_list.html', {'model_metrics': model_metrics, 'active_model': active_model,
                                               'error_message': error_message})


# Function that displays the registration page/form and registers a User to the system.
# Takes in request. Returns a redirection to homepage if the registration was successful
# Returns the registration page, an empty form, and an error message if the registration failed
# Author: Sandra Smoler Eisenberg
def register_user(request):
    form = UserCreationForm()
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            # Store the user in the DB and reference it so that we can login that user
            user = form.save()
            messages.success(request, 'Success')
            # Login the user
            login(request, user)
            return redirect('/')
        else:
            # If the form was not valid, get information of why the form was invalid to return to the UI
            errors = form.errors
            return render(request, 'register.html', {'form': form, 'errors': errors})

    return render(request, 'register.html', {'form': form})


# Function that displays the login page/form and logs in a user
# Takes in request. Returns a redirection to home page if login successful
# Returns the login page, with empty form and error message if login was not successful
# Author: Sandra Smoler Eisenberg
def login_user(request):
    error = ''
    if request.method == "POST":
        login_form = AuthenticationForm(request, data=request.POST)
        if login_form.is_valid():
            username = login_form.cleaned_data.get('username')
            password = login_form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            # If a user with the provided username and password exist - login the user
            if user is not None:
                login(request, user)
                return redirect('/')
            else:
                error = 'Invalid username or password'
        else:
            error = 'Invalid username or password'
    login_form = AuthenticationForm()
    return render(request, 'login_user.html', {'login_form': login_form, 'error': error})


# Function that logs out a logged in user
# Takes in request. Return a redirection to the homepage
# Author: Sandra Smoler Eisenberg
def logout_user(request):
    # Check whether user is logged in
    authenticated = authenticate_user(request)
    # If the end-user is logged in, we log them out and redirect them to the homepage
    if authenticated:
        logout(request)
        return redirect('/')
    # If a non-logged in end-user tries to access the /logout_user page, we redirect them to the homepage
    else:
        return redirect('/')


# Function to display history of uploaded images and their respective highest predicted class
# Takes in request. Returns the history page, image data and prediction data
# Author: Sandra Smoler Eisenberg
def show_history(request):
    # Check whether user is logged in
    authenticated = authenticate_user(request)
    # Only allow access to /history url if user is logged in
    if authenticated:
        user_history = history_controller.get_history_for_user(request.user)

        # Method gets triggered when the user hits the 'Delete' button for an image.
        if 'delete_image' in request.POST:
            # gets the id connected to the UserImage when hitting the delete button
            image_id = request.POST.get('delete_image')
            # deletes the user_image
            history_controller.delete_user_image(image_id)
            # get new user_history without the deleted item
            user_history = history_controller.get_history_for_user(request.user)

        return render(request, 'history.html', {'user_history': user_history})
    else:
        return redirect('/')


# Function to check whether end-user is logged in
# Takes in request. Returns True if end-user is logged in, else False
# Author: Sandra Smoler Eisenberg
def authenticate_user(request):
    if request.user.is_authenticated:
        authenticated = True
    else:
        authenticated = False
    return authenticated
