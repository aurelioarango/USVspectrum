#!/usr/bin/env python
# -*- coding: utf-8 -*-

#import TrainImageModel
from TestImageModel import *


def main(test_dataset_path, model_path):
    #print("main",model_path)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(device)
    test_data = load_test(test_dataset_path)
    if not test_data:
        print("No data")
    else:
        model = load_model(model_path)
        evaluate(test_data, model)

