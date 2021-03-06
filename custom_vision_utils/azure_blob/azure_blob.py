"""Utilities for working with Azure blob storage."""
import os
from typing import Optional, List

from azure.storage.blob import BlobClient, BlobServiceClient
from azure.storage.blob._models import BlobProperties


def set_connection_str_from_env_if_missing(connection_string: Optional[str]) -> str:
    """Set the connection string

    If the connection string is missing it is set from the environment variable STORAGE_CONNECTION_STRING.
    """
    if not connection_string:
        try:
            connection_string = os.environ["STORAGE_CONNECTION_STRING"]
        except KeyError:
            raise ValueError(
                "If `connection_string`is None, the environment variable STORAGE_CONNECTION_STRING must be set."
            )
    return connection_string


def get_blob(
    container_name: str, blob_name: str, connection_string: Optional[str] = None
) -> BlobClient:
    connection_string = set_connection_str_from_env_if_missing(connection_string)
    return BlobClient.from_connection_string(
        conn_str=connection_string,
        container_name=container_name,
        blob_name=blob_name,
    )


def get_blob_service_client(
    connection_string: Optional[str] = None,
) -> BlobServiceClient:
    """Get a blob service client.

    :param connection_string: Connection string for storage account.
    If none is supplied, it will be loaded from the environment variable STORAGE_CONNECTION_STRING.
    :return: BlobServiceClient
    """
    connection_string = set_connection_str_from_env_if_missing(connection_string)
    return BlobServiceClient.from_connection_string(connection_string)


def download_blob(blob, local_path):
    with open(local_path, "wb") as file:
        blob_data = blob.download_blob()
        file.write(blob_data.content_as_bytes())


def upload_to_blob(blob, local_path):
    with open(local_path, "rb") as data:
        blob.upload_blob(data)


def list_blobs(
    container_name: str,
    name_starts_with: Optional[str] = None,
    extensions: Optional[List[str]] = None,
    connection_string: Optional[str] = None,
) -> List[BlobProperties]:
    """List blobs in container

    Returned blobs are filtered on name_starts_with and extensions.

    :param container_name: Name of container
    :param name_starts_with: Only return blobs with names that tart with name_starts_with
    :param extensions: Only return blobs with names ends with one of the extensions in extensions.
    :param connection_string: Connection string for storage account.
    If none is supplied, it will be loaded from the environment variable STORAGE_CONNECTION_STRING.
    :return: list of blobs
    """
    blob_service_client = get_blob_service_client(connection_string=connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    blob_properties = container_client.list_blobs(name_starts_with=name_starts_with)
    blob_properties_filtered = []
    if extensions:
        for extension in extensions:
            blob_properties_filtered += [
                blob for blob in blob_properties if blob.name.endswith(extension)
            ]

    else:
        return [blob for blob in blob_properties]
    return blob_properties_filtered
