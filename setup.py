from setuptools import setup, find_packages

setup(
    name="cogmenta_core",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'tqdm',
        'huggingface_hub'
    ]
)
