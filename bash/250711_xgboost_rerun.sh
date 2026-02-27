#!/bin/bash
# I'm running this because I need the predictions for each of the folds

# Hing features
python3 src/get_results_xgboost.py --hing_features True\
                                   --alberta_features False\
                                   --alberta_score False\
                                   --feature_selection True\
                                   --n_features_to_select 100

# Alberta features
python3 src/get_results_xgboost.py --hing_features False\
                                   --alberta_features True\
                                   --alberta_score False
# Alberta score
python3 src/get_results_xgboost.py --hing_features False\
                                   --alberta_features False\
                                   --alberta_score True
