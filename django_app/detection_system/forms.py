"""
Original author(s): Sandra Smoler Eisenberg
Modified by: Linus Åberg, Johann Tammen, Linus Ivarsosn

File purpose: Default Django file. Defines forms to accept end-user input
"""
# Imports
from django import forms
from .models import UserImage


# Form for the Image model
# Author: Sandra Smoler Eisenberg
class ImageForm(forms.ModelForm):
    class Meta:
        model = UserImage
        fields = ['image']


# Form to upload csv file
# Author: Linus Åberg - guslinuab@student.gu.se
class csv_form(forms.Form):
    class Meta:
        title = forms.CharField(max_length=50)
        file = forms.FileField()
        description = forms.CharField(max_length=200)
