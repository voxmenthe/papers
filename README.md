# papers

A Python toolkit for managing and analyzing arXiv research papers.

## Features

- **arxiv-ocr**: Download papers from arXiv and extract their text using Mistral OCR
  - Convert papers to markdown or HTML format
  - Extract and save images from papers
  - Process local PDF files or download directly from arXiv
  
## Usage

### arxiv-ocr

```bash
# Process a paper from arXiv
python src/scripts/arxiv_ocr.py https://arxiv.org/abs/1706.03762

# Process a local PDF file
python src/scripts/arxiv_ocr.py --file-path path/to/paper.pdf

# Generate HTML output with the first 5 pages
python src/scripts/arxiv_ocr.py https://arxiv.org/abs/1706.03762 --pages 5 --html
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/papers.git
cd papers

# Install dependencies using Poetry
./project_setup.sh
```

## Requirements

- Python 3.12+
- Poetry
- Mistral API key
