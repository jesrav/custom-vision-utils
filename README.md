# Utility package for working with Azure Custom Vision
The package contains a number of utilities for working with [Azure Custom Vision](https://docs.microsoft.com/en-us/azure/cognitive-services/custom-vision-service/).

**The main benefits are:**
- Specify your Custom Vision projects in a yaml configuration file.
- Specify your image data sets in a yaml configuration file.
    - Image data sets can be stored locally or in Azure Blob storage.
    - Image data sets contain information about image file location and tags/ regions for classification and object detection tasks.
- Use a cli to do stuff like creating projects, training models, uploading images, exporting images and exporting models.
- When uploading/exporting images, keep a reference to the id of each image, using Custom Visions meta data. 

# Get started

## Requirements
Python >= 3.6

## Install
````bash 
pip install git+https://github.com/jesrav/custom-vision-utils.git
````

