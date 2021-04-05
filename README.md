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

## Required environment variables
To use the package the following environment variables needs to be set. 
The command line tool will automatically look for a .env file and set them from that (See the .env_example file).
````dotenv
# Azure Custom Vision environment variables
ENDPOINT=
TRAINING_KEY=
PREDICTION_KEY=
PREDICTION_RESOURCE_ID=

# Azure blob storage environment variable(Only required if you use blob storage).
# If you wish to set the storage connection string in the code, when initializing image data set objects, you can do that.
STORAGE_CONNECTION_STRING=
````

# Example usage

## Creating Custom Vision projects
Define your projects in yaml and create the projects
````yaml
projects:

  - project_name:  "my-classifier-project"
    domain_type: "Classification"
    domain_name: "General (compact)"
    classification_type: "Multiclass"
    publish_name: "my-classifier-model"
    tags:
      - name: "dunk"
        type: "Regular"
      - name: "layup"
        type: "Regular"

  - project_name: "my-object-detection-project"
    domain_type: "ObjectDetection"
    domain_name: "General (compact)"
    publish_name: "my-object-detection-model"
    tags:
      - name: "region"
        type: "Regular"
````
Create the projects using the cli
```bash
    cvis create-projects <path-to-yaml-config-file>
```

## Working with classification data
Define a classification dataset in yaml
```yaml
# Local data set
images:
- tag_names:
  - positive
  uri: data/images/image1.jpg
- tag_names:
  - negative
  uri: data/images/image2.jpg
- tag_names:
  - positive
  uri: data/images/image3.jpg
```
```yaml
# Blob storage data set
images:
- tag_names:
  - positive
  uri: data/images/image1.jpg
  container_name: my-container
- tag_names:
  - negative
  uri: data/images/image2.jpg
  container_name: my-container
- tag_names:
  - positive
  uri: data/images/image3.jpg
  container_name: my-container
```
Upload the images and tags to a Custom Vision project
```bash
    cvis upload-images <project-name> <path-to-yaml-config-file>
```
Load the image data set in your Python code
```Python
from custom_vision_utils.image_dataset import LocalClassifierDataSet
local_classifier_data_set = LocalClassifierDataSet.from_config("<path-to-yaml-config-file>")
```