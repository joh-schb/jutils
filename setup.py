from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent
install_requires = (this_directory / "requirements.txt").read_text().splitlines()
long_description = (this_directory / "README.md").read_text()

setup(
    name="jutils",
    version="0.9.1.1",
    packages=find_packages(),
    url="https://github.com/joh-fischer/jutils",
    license="MIT License",
    author="Johannes S. Fischer",
    description="Collection of useful utility functions.",
    install_requires=install_requires,
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
    ],
)
