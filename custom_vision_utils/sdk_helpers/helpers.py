import os
import tempfile
import zipfile
import time
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
from custom_vision_utils.classification.result import ImageClassifierResultResult
from custom_vision_utils.object_detection.result import ObjectDetectionResult


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


def get_training_credentials():
    return ApiKeyCredentials(in_headers={"Training-key": get_training_key_from_env()})


def get_prediction_credentials():
    return ApiKeyCredentials(
        in_headers={"Prediction-key": get_prediction_key_from_env()}
    )


def get_trainer():
    return CustomVisionTrainingClient(
        get_endpoint_from_env(), get_training_credentials()
    )


def get_predictor():
    return CustomVisionPredictionClient(
        get_endpoint_from_env(), get_prediction_credentials()
    )


def get_domain_id(trainer, name, type="Classification"):
    domain_id = [
        domain.id
        for domain in trainer.get_domains()
        if domain.name == name and domain.type == type
    ][0]
    if len(domain_id) == 0:
        raise KeyError(f"No domain with name {name}")
    else:
        return domain_id


def get_project_names(trainer):
    return [p.name for p in trainer.get_projects()]


def get_project_id(trainer, project_name):
    project_ids = [
        project.id for project in trainer.get_projects() if project.name == project_name
    ]
    if len(project_ids) == 0:
        raise KeyError(f"No project with name {project_name}")
    else:
        return project_ids[0]


def get_iteration_id(trainer, project_name, iteration_name):
    project_id = get_project_id(trainer, project_name)
    iterations = trainer.get_iterations(project_id)
    try:
        return [iter for iter in iterations if iter.name == iteration_name][0]
    except IndexError:
        raise ValueError(
            f"No iteration called {iteration_name}, "
            f"iterations available: {[iter.name for iter in iterations]}"
        )


def get_latest_iteration(iterations):
    return sorted(iterations, key=lambda x: x.created, reverse=True)[0]


def get_predictions_sorted_by_probability(
    image_prediction: Union[
        List[ObjectDetectionResult], List[ImageClassifierResultResult]
    ]
) -> List[ObjectDetectionResult]:
    return sorted(image_prediction, key=lambda x: x.probability, reverse=True)


def get_highest_proba_prediction(
    image_prediction: Union[
        List[ObjectDetectionResult], List[ImageClassifierResultResult]
    ]
) -> ObjectDetectionResult:
    return get_predictions_sorted_by_probability(image_prediction)[0]


def get_predicted_tags(
    image_prediction: Union[
        List[ObjectDetectionResult], List[ImageClassifierResultResult]
    ],
    prob_thr=0.5,
):
    return [pred.tag_name for pred in image_prediction if pred.probability > prob_thr]


def get_predicted_tag(image_prediction):
    return get_highest_proba_prediction(image_prediction).tag_name


def crop_image_based_on_object_detection(
    image: Image,
    object_detection_result: ObjectDetectionResult,
    threshold_width_crop: float,
    crop_padding: float,
) -> Image:
    """Crop pillow_utils based on object detection custom_vision_results.

    If The cropped regiona region is wider than a threshold, the pillow_utils is not cropped
    Padding is added to the cropped pillow_utils.

    :param image: Pillow pillow_utils
    :param object_detection_result: Object detection result
    :param threshold_width_crop: If the crop width is over threshold_width_crop,
    the pillow_utils is not cropped. Between 0 and 1.
    :param crop_padding: Padding added to crop. Between 0 and 1.
    :return:
    """
    bbox = object_detection_result.bounding_box
    if bbox.width < threshold_width_crop:
        cropped_image = crop_image(
            image,
            max(bbox.left - (crop_padding / 2), 0),
            max(bbox.top - (crop_padding / 2), 0),
            min(
                bbox.width + crop_padding,
                1 - (crop_padding / 2),
            ),
            min(
                bbox.height + crop_padding,
                1 - (crop_padding / 2),
            ),
        )
        return cropped_image
    return image


def download_model_iteration_as_tensorflow(
    project_name: str, out_model_folder: zipfile.Path, iteration_id: str = None
) -> None:
    """Download model iteration.

    If no iteration_id is passed the latest model iteration is downloaded.

    :param project_name:
    :param out_model_folder:
    :param iteration_id:
    :return:
    """

    if not out_model_folder.exists():
        os.makedirs(str(out_model_folder))

    trainer = get_trainer()
    project_id = get_project_id(trainer, project_name)
    if iteration_id is None:
        iterations = trainer.get_iterations(project_id)
        iteration_id = get_latest_iteration(iterations).id

    try:
        trainer.export_iteration(
            project_id=project_id,
            iteration_id=iteration_id,
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
        iteration_id=iteration_id,
    )[0].download_uri

    r = requests.get(uri, allow_redirects=True)
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(r.content)

        with zipfile.ZipFile(tmp, "r") as zip_ref:
            zip_ref.extractall(str(out_model_folder))


def azure_image_prediction_to_image_classifier_results(
    azure_imager_prediction: ImagePrediction,
) -> List[ImageClassifierResultResult]:
    image_classifier_results = [
        ImageClassifierResultResult(
            tag_name=prediction["tag_name"], probability=prediction["probability"]
        )
        for prediction in azure_imager_prediction.as_dict()["predictions"]
    ]
    return image_classifier_results


def api_classification(
    image, trainer, predictor, project_name, iteration_name
) -> List[ImageClassifierResultResult]:
    project_id = get_project_id(trainer, project_name)

    results = predictor.classify_image(
        project_id=project_id,
        published_name=iteration_name,
        image_data=pil_image_to_byte_array(image),
    )
    return azure_image_prediction_to_image_classifier_results(results)


def get_tag_dict(trainer, project_id):
    return {tag.name: tag.id for tag in trainer.get_tags(project_id)}


def get_tag_id(tag_name: str, trainer, project_id) -> str:
    tag_dict = get_tag_dict(trainer, project_id)
    try:
        return tag_dict[tag_name]
    except KeyError:
        raise KeyError(
            f"Tag `{tag_name}` not part of project tags. Allowed values are: {tag_dict.keys()}"
        )


def download_custom_vision_image(custom_vision_image, file_handler) -> None:
    """Download a custom vision image.

    :param custom_vision_image: A custom vision image object
    :param file_handler: File like object
    """
    r = requests.get(custom_vision_image.original_image_uri, allow_redirects=True)
    file_handler.write(r.content)
