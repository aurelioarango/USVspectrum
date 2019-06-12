#!/usr/bin/env python
# -*- coding: utf-8 -*-

#import TrainImageModel

from TestImageModel import *
#import sys
#sys.path.append(os.getcwd() + "/scripts/usv_sort")

import ClassifyImages


def main(test_dataset_path, model_path,out_path):

    """test_data = load_test(test_dataset_path)
    if not test_data:
        print("No data")
    else:
        model = load_model(model_path)
        evaluate(test_data, model)"""
    test_dataset_path
    ClassifyImages.main('predictions.txt',test_dataset_path,out_path)
    print(test_dataset_path)


