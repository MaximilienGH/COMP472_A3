"""
File name:     main.py
Authors:       Maximilien Fathi
Date:          December 13, 2020
Description:   Driver file used to run the whole program.
"""

import data_export
from OV_model import OV_model
from FV_model import FV_model
import pandas as pd
import time

train_df = pd.read_csv("Input_Files/covid_training.tsv", sep='\t') # , header=None
test_df = pd.read_csv("Input_Files/covid_test_public.tsv", sep='\t', header=None)

def run_model(model, model_type):
    """Runs a specific model and then exports results."""
    start = time.perf_counter()
    model.create_vocabulary()
    model.group_classes()
    model.count_words()
    model.create_probabilities_dict()
    end = time.perf_counter()
    print(f"Time for OV model {end - start:0.4f} seconds")
    data_export.create_trace_file(test_df, model, model_type)
    data_export.create_evaluation_file(test_df, model, model_type)

def main():
    """Starts entire program."""
    OV_clf = OV_model(train_df)
    run_model(OV_clf, "OV")
    
    FV_clf = FV_model(train_df)
    run_model(FV_clf, "FV")
    
if __name__ == "__main__":
    main()
