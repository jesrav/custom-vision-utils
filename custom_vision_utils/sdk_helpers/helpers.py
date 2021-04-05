import os
import tempfile
import time
import zipfile
from pathlib import Path
from typing import List, Union

import requests
from PIL import Image
from azure.cognitiveservices.vision.customvision.prediction.models import (
    ImagePrediction,
)
from azure.cognitiveservices.vision.customvision.training.models._models_py3 import (
    CustomVisionErrorException,
)
from azure.cognitiveservices.vision.customvision.training import (
    CustomVisionTrainingClient,
)
from azure.cognitiveservices.vision.customvision.prediction import (
    CustomVisionPredictionClient,
)
from msrest.authentication import ApiKeyCredentials

from custom_vision_utils.pillow_utils import pil_image_to_byte_array, crop_image
from custom_vision_utils.classification.result import ImageClassifierResult
from custom_vision_utils.object_detection.result import ObjectDetectionResult


ImageResult = Union[ObjectDetectionResult, ImageClassifierResult]
ImageResults = Union[List[ObjectDetectionResult], List[ImageClassifierResult]]


def get_training_key_from_env() -> str:
    """Get the training key from environment variable TRAINING_KEY"""
    try:
        training_key = os.environ["TRAINING_KEY"]
    except KeyError:
        raise ValueError("The environment variable TRAINING_KEY must be set.")
    return training_key


def get_endpoint_from_env() -> str:
    """Get the endpoint key from environment variable ENDPOINT"""
    try:
        training_key = os.environ["ENDPOINT"]
    except KeyError:
        raise ValueError("The environment variable ENDPOINT must be set.")
    return training_key


def get_prediction_key_from_env() -> str:
    """Get the endpoint key from environment variable ENDPOINT"""
    try:
        prediction_key = os.environ["PREDICTION_KEY"]
    except KeyError:
        raise ValueError("The environment variable PREDICTION_KEY must be set.")
    return prediction_key


def get_training_credentials() -> ApiKeyCredentials:
    return ApiKeyCredentials(in_headers={"Training-key": get_training_key_from_env()})


def get_prediction_credentials() -> ApiKeyCredentials:
    return ApiKeyCredentials(
        in_headers={"Prediction-key": get_prediction_key_from_env()}
    )


def get_trainer() -> CustomVisionTrainingClient:
    """Get Custom Vision training client."""
    return CustomVisionTrainingClient(
        endpoint=get_endpoint_from_env(),
        credentials=get_training_credentials()
    )


def get_predictor() -> CustomVisionPredictionClient:
    """Get Custom Vision prediction client."""
    return CustomVisionPredictionClient(
        endpoint=get_endpoint_from_env(),
        credentials=get_prediction_credentials()
    )


def get_domain_id(
    trainer: CustomVisionTrainingClient, domain_name, domain_type="Classification"
) -> str:
    """Get Custom Vision domain id.

    :param trainer: Custom Vision training client
    :param domain_name: Name of domain
    :param domain_type: Type of domain
    :return: Id of Custom Vision domain
    """
    domain_id = [
        domain.id
        for domain in trainer.get_domains()
        if domain.name == domain_name and domain.type == domain_type
    ][0]
    if len(domain_id) == 0:
        raise KeyError(f"No domain with name {domain_name}")
    else:
        return domain_id


def get_project_names(trainer: CustomVisionTrainingClient) -> List[str]:
    """Get Custom Vision project names

    :param trainer: Custom Vision training client
    :return: List of Custom Vision project names
    """
    return [p.name for p in trainer.get_projects()]


def get_project_id(trainer: CustomVisionTrainingClient, project_name: str) -> str:
    """Get Custom Vision project id

    :param trainer: Custom Vision training client
    :param project_name: Name of project
    :return: Custom Vision project id
    """
    project_ids = [
        project.id for project in trainer.get_projects() if project.name == project_name
    ]
    if not project_ids:
        raise KeyError(f"No project with name {project_name}")
    else:
        return project_ids[0]


def get_iteration(
    trainer: CustomVisionTrainingClient, project_name: str, iteration_name: str
):
    """Get custom vision model iteration.

    :param trainer: Custom Vision training client
    :param project_name: Name of project
    :param iteration_name: Name of iteration
    :return:
    """
    project_id = get_project_id(trainer, project_name)
    iterations = trainer.get_iterations(project_id)
    try:
        return [iteration for iteration in iterations if iteration.name == iteration_name][0]
    except IndexError:
        raise ValueError(
            f"No iteration called {iteration_name}, "
            f"iterations available: {[iteration.name for iteration in iterations]}"
        )


def get_latest_iteration(iterations):
    return sorted(iterations, key=lambda x: x.created, reverse=True)[0]


def get_results_sorted_by_probability(image_results: ImageResults) -> ImageResults:
    return sorted(image_results, key=lambda x: x.probability, reverse=True)


def get_highest_proba_result(image_results: ImageResults) -> ImageResult:
    return get_results_sorted_by_probability(image_results)[0]


def get_predicted_tag(image_classifier_results: List[ImageClassifierResult]) -> str:
    return get_highest_proba_result(image_classifier_results).tag_name


def get_tag_dict(trainer: CustomVisionTrainingClient, project_id: str):
    """Get dictionary mapping tag nane to tag id for a Custom Vision project.

    :param trainer: Custom Vision training client
    :param project_id: Project id
    :return: Dictionary mapping tag nane to tag id
    """
    return {tag.name: tag.id for tag in trainer.get_tags(project_id)}


def get_tag_id(
        tag_name: str, trainer: CustomVisionTrainingClient, project_id: str
) -> str:
    """Get tag id for Custum Vision project

    :param tag_name: Tag Name
    :param trainer: Custom Vision training client
    :param project_id: Project id
    :return: Tag id
    """
    tag_dict = get_tag_dict(trainer, project_id)
    try:
        return tag_dict[tag_name]
    except KeyError:
        raise KeyError(
            f"Tag `{tag_name}` not part of project tags. Allowed values are: {tag_dict.keys()}"
        )


def crop_image_based_on_object_detection(
    image: Image,
    object_detection_result: ObjectDetectionResult,
    threshold_width_crop: float,
    crop_padding: float,
) -> Image:
    """Crop pillow_utils based on object detection custom_vision_results.

    If the cropped region is wider than a threshold, the image is not cropped
    Padding is added to the cropping..

    :param image: Pillow image
    :param object_detection_result: Object detection result
    :param threshold_width_crop: If the crop width is over threshold_width_crop,
    the image is not cropped. Between 0 and 1.
    :param crop_padding: Padding added to crop. Between 0 and 1.
    :return:
    """
    bbox = object_detection_result.bounding_box
    if bbox.width < threshold_width_crop:
        return crop_image(
            image=image,
            xmin=max(bbox.left - (crop_padding / 2), 0),
            ymax=max(bbox.top - (crop_padding / 2), 0),
            width=min(
                bbox.width + crop_padding,
                1 - (crop_padding / 2),
            ),
            height=min(
                bbox.height + crop_padding,
                1 - (crop_padding / 2),
            ),
        )
    return image


def download_model_iteration_as_tensorflow(
    project_name: str, out_model_folder: Path, iteration: str = None
) -> None:
    """Download model iteration.

    If no iteration_id is passed the latest model iteration is downloaded.

    :param project_name: Name of Custum Vision project
    :param out_model_folder: Out folder for exported model
    :param iteration: Name of model iteration
    """

    if not out_model_folder.exists():
        os.makedirs(str(out_model_folder))

    trainer = get_trainer()
    project_id = get_project_id(trainer, project_name)
    if iteration is None:
        iterations = trainer.get_iterations(project_id)
        iteration = get_latest_iteration(iterations).id

    try:
        trainer.export_iteration(
            project_id=project_id,
            iteration_id=iteration,
            platform="TensorFlow",
            flavor="TensorFlowNormal",
        )
    # Model already queued for exporting.
    except CustomVisionErrorException:
        pass

    # Sleeping to allow azure to que the model for exporting
    time.sleep(5)

    trainer = get_trainer()
    uri = trainer.get_exports(
        project_id=project_id,
        iteration_id=iteration,
    )[0].download_uri

    r = requests.get(uri, allow_redirects=True)
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(r.content)

        with zipfile.ZipFile(tmp, "r") as zip_ref:
            zip_ref.extractall(str(out_model_folder))


def azure_image_prediction_to_image_classifier_results(
    azure_image_prediction: ImagePrediction,
) -> List[ImageClassifierResult]:
    """Get ImageClassifierResult from Azure Custum Vision prediction

    :param azure_image_prediction: Azure Custum Vision image prediction object.
    :return: ImageClassifierResult
    """
    return [
        ImageClassifierResult(
            tag_name=prediction["tag_name"], probability=prediction["probability"]
        )
        for prediction in azure_image_prediction.as_dict()["predictions"]
    ]


def api_classification(
    image: Image,
    trainer: CustomVisionTrainingClient,
    predictor: CustomVisionPredictionClient,
    project_name: str,
    iteration_name: str,
) -> List[ImageClassifierResult]:
    """Predict image class/classes using Custum Vision API.

    :param image: Pillow image
    :param trainer: Custom Vision training client
    :param predictor: Custom Vision prediction client
    :param project_name: Name of custum vision project
    :param iteration_name: Name of model iteration
    :return: List of image classification results
    """
    project_id = get_project_id(trainer, project_name)

    results = predictor.classify_image(
        project_id=project_id,
        published_name=iteration_name,
        image_data=pil_image_to_byte_array(image),
    )
    return azure_image_prediction_to_image_classifier_results(results)


def download_custom_vision_image(custom_vision_image, file_handler) -> None:
    """Download a custom vision image.

    :param custom_vision_image: A custom vision image object
    :param file_handler: File like object
    """
    r = requests.get(custom_vision_image.original_image_uri, allow_redirects=True)
    file_handler.write(r.content)
