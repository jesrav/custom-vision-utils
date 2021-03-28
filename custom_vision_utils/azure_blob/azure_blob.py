import os
from typing import Optional, Iterable

from azure.storage.blob import BlobClient, BlobServiceClient
from azure.storage.blob._models import BlobProperties


def set_connection_str_from_env_if_missing(connection_string: str) -> str:
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
    """Get an azure blob client"""
    connection_string = set_connection_str_from_env_if_missing(connection_string)
    return BlobClient.from_connection_string(
        conn_str=connection_string,
        container_name=container_name,
        blob_name=blob_name,
    )


def get_blob_service_client(
    connection_string: Optional[str] = None,
) -> BlobServiceClient:
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
    ends_with: Optional[str] = None,
    connection_string: Optional[str] = None,
) -> Iterable[BlobProperties]:
    blob_service_client = get_blob_service_client(connection_string=connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    blob_properties = container_client.list_blobs(name_starts_with=name_starts_with)
    if ends_with:
        return [blob for blob in blob_properties if blob.name.endswith(".jpg")]
    else:
        return blob_properties
