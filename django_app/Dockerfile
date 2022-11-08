#Specify use the official Python image as baseimage
FROM python:3.6

#Instruct docker not to buffer output from Python and instead send it to the terminal
ENV PYTHONBUFFERED 1

# RUN pip install Pillow

#Create a directory
RUN mkdir /app

#Set working directory
WORKDIR /app

#Copy all files from local project into container
ADD . /app

#Runs pip install for all specified packages
RUN pip install -r requirements.txt


