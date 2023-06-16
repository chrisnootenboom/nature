from setuptools import setup, find_packages

setup(
    author="Chris Nootenboom",
    description="helper functions for geospatial workflows",
    name="nature",
    version="0.1.0",
    packages=find_packages(include=["nature", "nature.*"]),
)
