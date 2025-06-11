# rt-rename

## Overview
Rename RT structure sets according to the TG263 guideline using Large Language Models.

This project provides a web-based interface built with Dash to facilitate the renaming of radiotherapy (RT) structure sets. It leverages Large Language Models (LLMs), both local (via Ollama) and cloud-based, to suggest standardized names based on the TG263 nomenclature guidelines. The application allows users to upload structure data in various formats (DICOM, CSV, or from a directory), review LLM-generated suggestions, and apply changes.

## Project Structure

- **`app.py`**: The main Dash application file. It defines the user interface, manages callbacks, and orchestrates the renaming workflow.
- **`utils.py`**: Contains core utility functions for:
    - Loading and parsing TG263 guidelines.
    - Reading and processing input structure files (DICOM, CSV, .nrrd).
    - Generating prompts for LLMs.
    - Interacting with LLM APIs.
    - Validating suggested names against the TG263 standard.
    - Updating DICOM files with new names.
- **`config/`**: Stores configuration files, including:
    - Prompt templates for different LLM interactions (e.g., `prompt_v1.txt`, `prompt_v2.txt`).
    - The `TG263_nomenclature.xlsx` file containing the official TG263 guidelines.
- **`docker/`**: Contains Docker configuration files (`compose.yaml`, `dockerfile`) for containerizing the application and its dependencies, including the Ollama LLM server.


## Workflow

1.  **Data Input**: Users can upload RT structure data through the web interface. Supported formats include DICOM RTstruct files, CSV files listing structure names, or by pointing to a directory containing structure files (e.g., .nrrd).
2.  **Parsing and Preparation**: The application parses the input data to extract the existing structure names.
3.  **LLM-Powered Renaming**: For each structure, a prompt is generated using prompt templates from the `config/` directory. This prompt, along with the TG263 guideline data, is sent to an LLM (configurable, can be local via Ollama or a cloud-based model).
4.  **Suggestion Review**: The LLM's suggested TG263 names, along with confidence scores (if applicable), are displayed in the web interface. The interface also allows users to verify the suggestions and accept or modify them.
5.  **Output Generation**:
    - The updated structure list can be exported (e.g., as a CSV).
    - If DICOM files were provided, the application can write the new names back into the DICOM RTSTRUCT files.

## Getting Started

### Prerequisites
- Python 3.8 or higher


## License
This project is licensed under the MIT License.