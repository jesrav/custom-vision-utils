from setuptools import setup, find_packages


setup(
    name="custom_vision_utils",
    version="0.0.1",
    description="Utilities for working with Azure Custom Vision.",
    author="Jes Ravnbøl",
    author_email="jesravnbol@hotmail.com",
    license='MIT',
    packages=find_packages(where="."),
    install_requires=[
        "Pillow",
        "azure-cognitiveservices-vision-customvision",
        "msrest",
        "azure-storage-blob",
        "azure-storage-file-datalake",
        "opencv-python",
        "imutils",
        "imgaug",
        "tensorflow",
        "pydantic",
        "pyyaml",
        "python-dotenv",
        "click",
        "rich",
    ],
    entry_points={
        "console_scripts": [
            "cvis = custom_vision_utils.cli.cli:cli",
        ]
    },
)
