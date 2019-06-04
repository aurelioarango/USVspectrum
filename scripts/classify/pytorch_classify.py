#!/usr/bin/env python
# -*- coding: utf-8 -*-

#import TrainImageModel
from TestImageModel import *


def main(test_dataset_path, model_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    test_data = load_test(test_dataset_path)
    model = load_model(model_path)
    evaluate(test_data, model)

