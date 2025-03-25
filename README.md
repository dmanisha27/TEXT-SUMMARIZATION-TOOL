# AI Text Summarization Tool

## Overview
This project provides multiple approaches to text summarization using Natural Language Processing (NLP) techniques. The tool supports:
- **NLTK Frequency-Based Summarization**
- **TF-IDF-Based Summarization**
- **Transformer-Based Summarization (Deep Learning)**

## Files
- `summarize.py`: Main script implementing different summarization techniques.
- `requirements.txt`: List of dependencies required for the project.
- `sample.txt`: Sample text file for testing summarization.

## Setup Instructions
### Prerequisites
Ensure you have Python installed (preferably Python 3.8+). Install dependencies using:
```sh
pip install -r requirements.txt
```

### Running the Tool
Execute the script:
```sh
python summarize.py
```
Follow the interactive prompts to input text and select a summarization method.

## Dependencies
The project requires the following Python libraries:
- `nltk`
- `scikit-learn`
- `numpy`
- `torch`
- `transformers`

## Features
- Tokenization and stopword removal using **NLTK**
- Text ranking with **TF-IDF vectorization**
- Deep learning-based summarization using **Hugging Face Transformers**
- GPU acceleration support for Transformer models (if CUDA is available)

## Author
Manisha

