# ComputerVisionProjects

Computer Vision project:

- On ambit of computer vision class the idea is to create a model to classify the position of my dog(Pilar) is.

## Structure

This repository will be used for:

- Train models
- Perform Inferences

This repository goal is to have three modules, pre-processing, training and classification.

## Setup Virtual Environment

Make sure you have `poetry` installed locally and all packages installed in the environment.

- To install `poetry` just use the `pip3` command: `pip3 install poetry`.
- To create the virtual environment, run: `poetry install`.
- To learn more about [Poetry](https://python-poetry.org/docs/).

## Input data generation

For model training we will use as input multiple video files:

We will use Pilar videos on diferent positions: seat, sleep, down, XXX.

### transform images:

- 'transform_images.py': we will pre proccess this data and transform in fotos

### transform_split_data:

-






**ATTENTION**:

`DO NOT use Github UI to create tags for the development branches.
The UI links the GIT tag with a release and releases are only done for QA and Production environments from the 'main' branch.`

To create and push a tag, use the commands bellow:

- Create: `git tag <version> <commit-sha>`
- Push: `git push origin <version>`

---
