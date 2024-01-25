# Write-similarity

## Overview

Write-similarity is a stylometry analysis tool designed to evaluate and compare the writing styles against known authors. Utilizing a blend of modern NLP techniques, including word embeddings, POS tagging, sentence transformers, and dependency trees, this project leverages an ensemble method approach for deep stylistic analysis. By generating synthetic data for each author through the OpenAI API, Write-similarity offers a novel way to study and understand authorial uniqueness in text.

built by built by [@psbagga17](https://github.com/psbagga17) and [@Naveen-Kannan](https://github.com/Naveen-Kannan)
email us at [psbagga17@gmail.com](mailto:psbagga17@gmail.com?subject=Write-Similarity) and [naveenkannan222@gmail.com](mailto:naveenkannan222@gmail.com?subject=Write-Similarity)



## Key Features

- **Synthetic Data Generation**: Uses the OpenAI endpoint to create synthetic texts mimicking the styles of 255 authors, providing a rich dataset for analysis. View all authors in the [authors.txt](data/authors.txt) file.
- **Word2Vec Models**: Employs Word2Vec for semantic similarity, capturing the contextual nuances of words across different authorial styles.
- **Syntax POS Tagging**: Integrates Part-Of-Speech tagging to analyze the syntactic patterns unique to each author.
- **Sentence Transformers**: Utilizes sentence-level embeddings to understand the deeper semantic structures within texts.
- **Dependency Tree Analysis**: Applies dependency parsing to examine the grammatical structures that characterize an author's writing.


Note that using syntheic data blurs the line between semantics and syntax. Still, using the OpenAI API to generate synthetic data is a novel approach to stylometry analysis, especially as the amount of non-human generated text increases. We are actively looking at ways to a) improve synthetic data generation and b) improve the overall accuracy of the model.

To augment performance, tweak the weight hyperparamters in the [output.py](output.py) file.

## Getting Started

To get started with Write-similarity, clone this repository to your local machine. Ensure you have Python 3.6+ installed along with the necessary libraries: `torch`, `transformers`, `nltk`, `gensim`, and `spacy`.

use the requirements.txt file to install the necessary libraries

## Usage
View the [output.py](output.py) file for intuitive usage. 

If you dont use any argparse arguments, wait until prompted, then pass in text. The program will then output the author with the highest similarity score.