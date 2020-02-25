### How to run

- python3 productranker.py

- The defaults are already setup which if the trained model is not found it starts to train model.
- You need to add data files in a folder from where the model finds file

- For Running baseline algorithm
    - python3 baseline_model.py


### Files

- project.ipynb - Project Writeup.
- productranker.py - To train Main Model
- cpsir.py - Neural Network which called in inside the product ranker python file.
- data_utils.py -  Utilities for data preprocessing.
- utils.py - Utilities used by both main model and the baseline model.
- baseline_model.py -  To train the baseline model.
- baseline_utils.py - Utilities specific to baseline model.
- requirements.txt