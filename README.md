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
The examples below will use images stores locally, but you could equally use Azure blob storage, 
as long as the [right environment variable are set](#Required-environment-variables).

## Creating Custom Vision projects
Define your projects in yaml.
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
Create the projects in your Custom Vision project using the cli
```bash
    cvis create-projects <path-to-yaml-config-file>
```

## Defining a classification dataset in yaml
This shows how to define a classification data set, but you can correspondingly define object detection data sets. 
```yaml
# Local data set
images:
- tag_names:
  - positive
  uri: data/images/image1.jpg
  name: image1
- tag_names:
  - negative
  uri: data/images/image2.jpg
  name: image2
- tag_names:
  - positive
  uri: data/images/image3.jpg
  name: image3
```
```yaml
# Blob storage data set
images:
- tag_names:
  - positive
  uri: data/images/image1.jpg
  name: image1
  container_name: my-container
- tag_names:
  - negative
  uri: data/images/image2.jpg
  name: image2
  container_name: my-container
- tag_names:
  - positive
  uri: data/images/image3.jpg
  name: image3
  container_name: my-container
```

## Uploading images and tags to a Custom Vision project
```bash
    cvis upload-images <project-name> <path-to-yaml-config-file>
```
## Working with the data sets in python code
You can load the image data set in your Python code
```Python
from custom_vision_utils.image_dataset import LocalClassifierDataSet
local_classifier_data_set = LocalClassifierDataSet.from_config("<path-to-yaml-config-file>")
```
## Example of processing the images and creating a new classification data set.
```Python
from pathlib import Path
from custom_vision_utils.image import LocalClassifierImage
from custom_vision_utils.image_dataset import LocalClassifierDataSet

LOCAL_PROCESSED_IMAGE_FOLDER = "<directory-path>"

# Initialize a new classifier dataset 
processed_classifier_data_set = LocalClassifierDataSet() 

# Loop over the classification images to do some proccessing. 
for image in local_classifier_data_set:
    proccesed_pillow_image = do_stuff_to_image(image.get_pil_image())
    processed__local_classification_image = LocalClassifierImage.from_pil_image(
        image=proccesed_pillow_image,
        uri=Path(LOCAL_PROCESSED_IMAGE_FOLDER) / Path(image.name + "_processed.jpg"),
        tag_names=image.tag_names,
        name=image.name,
    )
    processed_classifier_data_set.append(processed__local_classification_image)

# Write config for new local processed data set to yaml file
processed_classifier_data_set.write_config("<processed-config-path>")
```

## Exporting images and tags from a Custom Vision project
```bash
cvis export-images <project-name> <directory> --data-config-outpath <outpath-for-yaml-config-file>
```
If you uploaded the images using the cvis command line tool, the exported images will have the same names.
This is note the case if you just use the Azure Custom Vision SDK. Keeping the names allows you to use Custom Vision to do tagging or validating  
of classes or regions and linking the exporting tags/regions back to the original images.
