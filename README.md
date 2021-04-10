# Utility package for working with Azure Custom Vision
The package contains a number of utilities for working with [Azure Custom Vision](https://docs.microsoft.com/en-us/azure/cognitive-services/custom-vision-service/).
It wraps the Custom Vision Python SDK and add extra functionality.

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
Define your projects in yaml.
````yaml
projects:

  - project_name:  "my-classifier-project"
    domain_type: "Classification"
    domain_name: "General (compact)"
    classification_type: "Multiclass"
    publish_name: "my-classifier-model"
    tags:
      - name: "flower1"
        type: "Regular"
      - name: "flower2"
        type: "Regular"

  - project_name: "my-object-detection-project"
    domain_type: "ObjectDetection"
    domain_name: "General (compact)"
    publish_name: "my-object-detection-model"
    tags:
      - name: "region"
        type: "Regular"
````
Create the projects in your Azure Custom Vision resource using the cli
```bash
    cvis create-projects <path-to-yaml-config-file>
```

## Defining a classification dataset in yaml

### Classification data sets
```yaml
# Local classification data set
images:
- tag_names:
  - flower1
  - flower2
  uri: data/images/image1.jpg
  name: image1
- tag_names:
  - flower1
  uri: data/images/image2.jpg
  name: image2
```
```yaml
# Blob storage classification data set
images:
- tag_names:
  - flower1
  - flower2
  uri: data/images/image1.jpg
  name: image1
  container_name: my-container
- tag_names:
  - flower1
  uri: data/images/image2.jpg
  name: image2
  container_name: my-container
```
```yaml
# Local classification data set, where separate tags are stored in separate directories  
image_dirs:
  - path_dir: data/train/positive
    tag_names:
    - positive
  - path_dir: data/train/negative
    tag_names:
      - negative
```
```yaml
# Blob classification data set, where separate tags are stored in separate directories 
image_dirs:
  - container_name: my-container
    blob_dir: data/train/positive
    tag_names:
    - positive
  - container_name: my-container 
    blob_dir: data/train/negative
    tag_names:
      - negative
```
### Object detection data sets
```yaml
# Local object detection data set
images:
  - uri: data/images/image1.jpg
    regions:
      - bounding_box:
          left: 0.2
          top: 0.1
          width: 0.5
          height: 0.3
        tag_name: object_tag1
      - bounding_box:
          left: 0.5
          top: 0.1
          width: 0.1
          height: 0.1
        tag_name: object_tag2
  - uri: data/images/image2.jpg
    regions:
      - bounding_box:
          left: 0.2
          top: 0.2
          width: 0.2
          height: 0.50
        tag_name: object_tag2
```
```yaml
# Blob storage object detection data set
images:
  - uri: data/images/image1.jpg
    container_name: my-container
    regions:
      - bounding_box:
          left: 0.2
          top: 0.1
          width: 0.5
          height: 0.3
        tag_name: object_tag1
      - bounding_box:
          left: 0.5
          top: 0.1
          width: 0.1
          height: 0.1
        tag_name: object_tag2
  - uri: data/images/image2.jpg
    container_name: my-container
    regions:
      - bounding_box:
          left: 0.2
          top: 0.2
          width: 0.2
          height: 0.50
        tag_name: object_tag2
```
## Uploading images and tags to a Custom Vision project
To upload a classification or object detection data sets defined in a yaml config, you can use the cli. 
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
This is not the case if you just use the Azure Custom Vision SDK. 
Keeping the names allows you to use Custom Vision to do tagging or validating of classes or regions and linking the exporting tags/regions back to the original images.
