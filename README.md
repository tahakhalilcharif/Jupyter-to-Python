# Jupyter Notebook to Python Converter

A simple, efficient tool to convert Jupyter notebooks (.ipynb) into standard Python scripts (.py).

## Features

- **Code Preservation**: Maintains all code cells in their original order
- **Markdown Support**: Converts markdown cells to Python comments (optional)
- **Clean Formatting**: Preserves readability with proper spacing and formatting
- **Flexible Usage**: Can be used as a command-line tool or imported as a module

## Installation

Clone the repository:

```bash
git clone https://github.com/tahakhalilcharif/Jupyter-to-Python notebook-to-python
cd notebook-to-python
```

No additional dependencies are required beyond the Python standard library.

## Usage

### Command Line

```bash
# Basic usage - outputs to {notebook_name}.py
python jupyter_to_python.py my_notebook.ipynb

# Custom output path
python jupyter_to_python.py my_notebook.ipynb -o converted_script.py

# Skip markdown cells
python jupyter_to_python.py my_notebook.ipynb --no-markdown
```

### As a Module

You can also import and use the converter in your own Python scripts:

```python
from jupyter_to_python import convert_notebook_to_script

# Convert with default settings
convert_notebook_to_script('my_notebook.ipynb')

# Customize conversion
convert_notebook_to_script('my_notebook.ipynb', 
                          output_path='custom_name.py',
                          include_markdown=False)
```

## Example

Input notebook (`example.ipynb`):
```json
{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": ["# Example Notebook\n", "This is a simple example."]
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": ["import pandas as pd\n", "import numpy as np\n", "\n", "print('Hello, world!')"]
    }
  ],
  "metadata": {...},
  "nbformat": 4,
  "nbformat_minor": 4
}
```

Output script (`example.py`):
```python
# Converted from Jupyter notebook using notebook-to-python converter
# Original notebook: example.ipynb

# ============================================================
# MARKDOWN CELL
# ============================================================
# # Example Notebook
# This is a simple example.
# ============================================================

import pandas as pd
import numpy as np

print('Hello, world!')
```

## Why Use This Tool?

- **Version Control**: Better track changes in Git repositories
- **Script Integration**: Easily integrate notebook code into larger Python projects
- **Automation**: Use in CI/CD pipelines to convert notebooks to scripts
- **Code Review**: Simplified view of code without notebook metadata
- **Portability**: Run notebook code without requiring Jupyter

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by the need to easily integrate Jupyter notebook code with traditional Python workflows
- Thanks to the Jupyter project for their amazing notebook format