# Composer-Classification
Please following below steps:
Step 0: Change the config.py: set the input, output folders and the sub directories that have the midi files of different composers. In addition, please change the folder to save distance table and named chords encode path
Step 1: Go to the folder contain these file and use: python 01_create_named_chords_encode.py
Step 2: python 02_create_distance_table.py
Step 3: python 03_extract_features.py
Step 4: 04_run_machine_learning_methods.py
Step 5: 05_music21_standard_feature_compare.py to run the Logistic Regression and Linear Discriminant Analysis on the standard feature of Music21 in order to compare with proposed features from step 1 through step 4
