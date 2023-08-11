import argparse, sys

from src.data_loader import DataLoader
from src.preprocess import Preprocess
from src.data_augmentation import DataAugmentation
from src.models import RandomForest, GradientBoosting
from src.decorrelation import Decorrelator

import settings
import pandas as pd

import tensorflow as tf
import tensorflow_decision_forests as tfdf

import matplotlib.pyplot as plt


def parse_params(params):
    """ parse parameters """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/dataset.xlsx')
    parser.add_argument('--augmentation', type=bool, default=False)
    parser.add_argument('--training_split', type=float, default=0.8)
    parser.add_argument('--opt', type=str, default='div')
    parser.add_argument('--save', type=bool, default=False)
    parser.add_argument('--label', type=str, default='Patient addmited to intensive care unit (1=yes, 0=no)')
    parser.add_argument('--show', type=bool, default=False)
    parser.add_argument('--file_type', type=str, default='excel')
    parser.add_argument('--model', type=str, default='sklearn')
    parser.add_argument('--type_tree', type=str, default='forest')
    parser.add_argument('--tune', type=bool, default=False)
    parser.add_argument('--decorrelate', type=bool, default=False)

    return parser.parse_args(params)


def main(params):
    """ main """
    params = parse_params(params)

    # load data
    data_loader = DataLoader(params.data_path)
    train_data, test_data = data_loader.load(
        training_split=params.training_split,
        file_type=params.file_type,
    )

    # preprocess data
    preprocessor = Preprocess()

    tf_data = True 
    if params.model != 'tf' or params.augmentation:
        tf_data = False

    train_data = preprocessor.preprocess(
        train_data,
        label=params.label,
        categorical_cols=settings.categoricals,
        scale_cols=settings.scale_cols,
        drop_cols=settings.drop_cols,
        tf_data=tf_data)
    
    test_data = preprocessor.preprocess(
        test_data,
        label=params.label,
        categorical_cols=settings.categoricals,
        scale_cols=settings.scale_cols,
        drop_cols=settings.drop_cols,
        tf_data=tf_data)
        
    # decorrelate data
    if params.decorrelate:
        decorrelator = Decorrelator()

        train_data = decorrelator.fit_transform(train_data)
        # TODO :- separate fit transform for test data? is it right?
        test_data = decorrelator.transform(test_data)

    # data augmentation
    if params.augmentation:
        data_augmentation = DataAugmentation()

        train_data = data_augmentation.load(
            train_data,
            label=params.label,
            opt=params.opt,
            save=params.save
            )

        train_data = preprocessor.preprocess(
            train_data,
            label=params.label
            )
        
        test_data = data_augmentation.load(
            test_data,
            label=params.label,
            opt=params.opt,
            save=params.save
            )

        test_data = preprocessor.preprocess(
            test_data,
            label=params.label
            )

    # select tree model 
    if params.type_tree == 'forest':
            model = RandomForest
    elif params.type_tree == 'boosting':
        model = GradientBoosting
    else:
        raise ValueError('Model not correctly defined')
    
    # create model 
    model = model(type=params.model,)
    
    if params.tune:
        # build model and select cross validation parameters
        if params.model == 'sklearn':
            cv_params = settings.params_sklearn
        elif params.model == 'tf':
            cv_params = settings.params_tf
        elif params.model == 'xgboost':
            assert(params.type_tree == 'boosting')
            cv_params = settings.params_xgb
        else:
            raise ValueError('Model not correctly defined')
        
        # tune model
        model.tune(train_data, cv_params)

    # train model
    model.fit(train_data)

    # evaluate model
    print(model.test(test_data))

    # plot feature importance
    print(model.feature_importance())


    
if __name__ == '__main__':
    main(sys.argv[1:])