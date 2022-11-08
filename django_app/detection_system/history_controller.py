"""
Original author(s): Sandra Smoler Eisenberg (creating functionality), Johann Tammen (creating file, refactor to its
own file)
Modified by:

File purpose: Methods to control the user_images in the system & delete history items (user_images)
"""
import os.path
from . import variables
from .models import UserImage
from django.conf import settings


# Function to get the history of uploaded images for a logged in user
# Takes in the id of a user
# Output: zip of image data, filename, prediction, info link and id
# Author: Sandra Smoler Eisenberg, (Johann Tammen)
def get_history_for_user(user):
    # Make all queries in descending order by id so we get the latest upload first in the lists
    # Query the images that belongs to the logged in user and store them as a list in
    user_images = list(UserImage.objects.filter(user=user).values_list('image', flat=True).order_by('-id'))
    # Query the image predictions that belongs to the logged in user and store them as a list
    image_predictions = list(UserImage.objects.filter(user=user).values_list('prediction', flat=True).order_by('-id'))
    # Query the image dates that belongs to the logged in user and store them as a list
    image_dates = list(UserImage.objects.filter(user=user).values_list('date', flat=True).order_by('-id'))
    # Query the image ids that belongs to the logged in user and store them as a list
    ids = list(UserImage.objects.filter(user=user).values_list('id', flat=True).order_by('-id'))
    # List to store the class label for the image predictions
    prediction_classes = []
    # List to store the links to the info about the classes
    class_links = []
    # Retrieve the class labels and class links for the classes predicted
    for label in image_predictions:
        prediction_classes.append(variables.LABEL_DIRECTORY[label])
        class_links.append(variables.LINK_DIRECTORY[label])

    # Zip all lists so they can be iterated over simultaneously in the HTML
    return zip(image_dates, user_images, prediction_classes, class_links, ids)


# Function to delete a user_image in the db and file system
# Takes in the id of a user_image
# Output: Delete image in db and file system
# Author: Johann Tammen
def delete_user_image(image_id):
    user_image = UserImage.objects.get(pk=image_id)
    file_name = user_image.image
    user_image.delete()
    os.remove(os.path.join(settings.MEDIA_ROOT, str(file_name)))
