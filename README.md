# RT-Rename

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-enabled-blue.svg)](https://www.docker.com/)

## Overview

RT-Rename is a web-based application designed to standardize radiotherapy (RT) structure nomenclature according to the TG263 guidelines using Large Language Models (LLMs). The application provides an intuitive interface for renaming RT structure sets, supporting both local models (via Ollama) and cloud-based models for intelligent structure name suggestions.

### Key Features

- 🏥 **TG263 Compliance**: Standardizes structure names according to international radiotherapy guidelines
- 🤖 **AI-Powered**: Uses state-of-the-art LLMs for intelligent name suggestions
- 📁 **Multiple Formats**: Supports DICOM RTstruct, CSV, and .nrrd files
- 🌐 **Web Interface**: User-friendly Dash-based interface
- 🐳 **Docker Ready**: Containerized deployment with Docker Compose
- 💾 **Export Options**: Export as CSV or updated DICOM files


## Project Structure


## Installation & Setup

### Option 1: Docker (Recommended)

#### Prerequisites
- Docker and Docker Compose
- NVIDIA GPU with Docker GPU support (optional, for local LLM models)

#### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/LMUK-RADONC-PHYS-RES/rt-rename.git
   cd rt-rename
   ```

2. **Start the application**
   ```bash
   cd docker
   docker-compose up -d
   ```

3. **Access the application**
   - Web interface: http://localhost:8055
   - Ollama API: http://localhost:11435

#### Environment Variables

If you want to use cloud inference create a `.env` file in the root directory for open-ai API compatible cloud-based models:

```properties
OPEN_AI_URL=your_api_url_here
OPEN_AI_API_KEY=your_api_key_here
```

### Option 2: Local Installation

#### Prerequisites
- Python 3.8 or higher
- pip package manager

#### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd rt-rename
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r docker/requirements.txt
   ```

4. **Install Ollama (for local models)**
   ```bash
   # On Linux/macOS
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # On Windows
   # Download from https://ollama.ai/download
   ```

5. **Start Ollama service**
   ```bash
   ollama serve
   ```

6. **Pull required models**
   ```bash
   ollama pull llama3.1:70b-instruct-q4_0
   # or other models as configured in config/models.json
   ```

7. **Run the application**
   ```bash
   python app.py
   ```

8. **Access the application**
   - Web interface: http://localhost:8055

## Usage Guide

### Web Interface

1. **Configure Settings**
   - Select nomenclature type (TG263 or TG263_reverse)
   - Choose anatomical regions (Thorax, Head and Neck, etc.)
   - Set target volume filtering
   - Select LLM model and prompt version

2. **Upload Structure Data**
   - DICOM RTstruct files (.dcm)
   - CSV files with structure names
   - Directory with .nrrd files (drag & drop supported)

3. **Review Suggestions**
   - View AI-generated TG263 name suggestions
   - Check confidence scores and verification status
   - Accept/reject individual suggestions
   - Add comments for manual review

4. **Export Results**
   - Export as CSV file
   - Export as updated DICOM RTstruct file

### Batch Processing

For large-scale processing, use the batch script:

```python
python batch_rename.py
```

Edit the script to specify:
- Input file paths
- Output file paths
- Model and prompt settings
- Batch size

### Supported File Formats

#### DICOM RTstruct
- Standard DICOM RT Structure Set files
- Automatically extracts ROI names
- Can export updated DICOM files

#### CSV Files
- First column: structure names
- Optional additional columns for existing TG263 names, confidence, etc.
- Exports enhanced CSV with AI suggestions

#### .nrrd Files
- Directory containing structure files
- Filters out intermediate files (_stitched, _s2_def, _s2)
- Supports batch processing

## Configuration

### Models Configuration

Edit `config/models.json` to add or modify available models:

```json
{
    "name": "Custom Model",
    "parameters": "7B",
    "model_str": "custom-model:7b",
    "cloud": false
}
```

### Prompt Engineering

Customize prompts in `config/prompt_v*.txt`:
- Add domain-specific instructions
- Modify output format requirements
- Include additional context or examples

### TG263 Guidelines

The `config/TG263_nomenclature.xlsx` file contains the official TG263 nomenclature. Update this file to use custom guidelines or newer versions.

## API Reference

### Core Functions

#### `run_model()`
```python
run_model(model, prompt, guideline, region, structure_dict, gui=True, uncertain=False)
```
Main function for processing structures with LLM.

#### `parse_csv()`, `parse_dicom()`
Parse input files into standardized structure format.

#### `update_dicom()`
Update DICOM files with new structure names.

## Troubleshooting

### Common Issues

1. **GPU Memory Issues**
   ```bash
   # Use smaller models or increase swap
   ollama pull llama3.2:3b-instruct-fp16
   ```

2. **Port Conflicts**
   ```bash
   # Change ports in docker/compose.yaml
   ports:
     - "8056:8055"  # Change external port
   ```

3. **Model Loading Errors**
   ```bash
   # Pull model manually
   docker exec -it rt-rename-ollama-1 ollama pull llama3.1:70b-instruct-q4_0
   ```

4. **File Upload Issues**
   - Check file permissions
   - Verify file format compatibility
   - Ensure sufficient disk space

### Performance Optimization

- Use GPU acceleration for local models
- Adjust batch sizes based on available memory
- Use cloud models for faster inference
- Enable caching for repeated operations

## Development

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Code formatting
black *.py
flake8 *.py
```

### Adding New Models

1. Update `config/models.json`
2. Implement model-specific handling in `utils.py`
3. Test with sample data
4. Update documentation

## Citation

If you use RT-Rename in your research, please cite:

```bibtex
@software{rt_rename_2025,
  title={RT-Rename: AI-Powered Radiotherapy Structure Standardization},
  author={ArtLab},
  year={2025},
  url={https://github.com/your-repo/rt-rename}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Acknowledgments

- TG263 Working Group for nomenclature standards
- Ollama team for local LLM infrastructure
- Dash/Plotly for the web framework
- PyDICOM developers for DICOM handling capabilities