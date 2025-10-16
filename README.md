# CR_radio_classifier

This github project uses Machine Learning to build a classifier that discriminates between cosmic ray signals and transient noise in the radio signals for the GP300 prototype of the GRAND experiment.

## Requirements

To run the full analysis, you need to have the `grand` module installed. It is need for extracting the datasets from the GRAND data/sims (directory `data_formating`). 

If you already have access to the correct datatsets, the `grand` module is not needed elsewhere.

## How to use

This was built to be a user-friendly page. To make it your own, you only need to change the two variables `master_path` and `classifier_path` in the `imported_fcts.py` function.

The first is used as an output directory (datasets and plots). The second should target this project's directory.

The directory `data_formating` contains all programs building the datasets needed for the training of the ML models.
Note that the `validation_1_datasets.py` program builds a validation dataset to check if the model reacts well to identified cosmic ray signals (tested in `model_testing/verif_1.py`).

The directory `model_training` contains all programs that define, rain and test the model.