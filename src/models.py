from typing import Union, Any, List, Optional

import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from xgboost import XGBClassifier

import tensorflow_decision_forests as tfdf


class AbstractTreeModel:
    """ Abstract class implementing tree-based models """
    def __init__(
            self, 
            type: str = 'sklearn',
            **kwargs: Optional[Any]
    ):
        """ 
        :param type: 
            str, type of model to use

        :param tune:
            bool, whether to tune hyperparameters

        :param kwargs:
            dict, keyword arguments for model
        """
        super(AbstractTreeModel, self).__init__()
        self.type = type
        self.kwargs = kwargs

    def tune(
            self,
            train_data: Union[pd.DataFrame, tf.data.Dataset],
            params: dict,
    ):
        """ tune hyperparameters """
        print(type(train_data))
        if self.type == 'tf':
            assert isinstance(train_data, tf.data.Dataset)
            tuner = tfdf.tuner.RandomSearch(
                num_trials=10
            )
            
            for key, value in params.items():
                tuner.choice(key, value)
            self.kwargs['tuner'] = tuner

        else:
            assert isinstance(train_data, pd.DataFrame)
            cv = GridSearchCV(
                estimator=self.model(),
                param_grid=params,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )

            cv.fit(
                train_data.drop('label', axis=1),
                train_data['label']
            )
            self.kwargs = cv.best_params_

    def fit(
            self,
            train_data: Union[pd.DataFrame, tf.data.Dataset],
    ):
        """ fit model """
        self.model = self.model(**self.kwargs)
        if self.type == 'tf':
            assert isinstance(train_data, tf.data.Dataset)
            self.model.compile(metrics=["accuracy"])
            self.model.fit(train_data)
        else:
            assert isinstance(train_data, pd.DataFrame)
            self.model.fit(
                train_data.drop('label', axis=1),
                train_data['label']
            )

    def test(
            self,
            test_data: Union[pd.DataFrame, tf.data.Dataset],
    ) -> Union[pd.DataFrame, float, List[float], dict]:
        """ test model """
        if self.type == 'tf':
            assert isinstance(test_data, tf.data.Dataset)
            return self.model.evaluate(test_data, return_dict=True)
        else:
            assert isinstance(test_data, pd.DataFrame)
            return self.model.score(
                test_data.drop('label', axis=1),
                test_data['label']
            )

    def feature_importance(
            self,
            metric: str = 'SUM_SCORE'
    ) -> List[Any]:
        """ feature importance """
        if self.type == 'tf':
            importance = self.model.make_inspector().variable_importances()
            if metric is not None:
                importance = importance[metric]
            return importance
        elif self.type == 'sklearn':
            importance = np.mean([
                tree.feature_importances_ 
                for tree in self.scores['estimator']],
                axis=0)
            return pd.Series(importance).sort_values(ascending=False)
        elif self.type == 'xgboost':
            return self.model.get_booster().get_score()


class RandomForest(AbstractTreeModel):
    """ Random Forest model """
    def __init__(
            self, 
            type: str = 'sklearn',
            **kwargs
    ):
        """ 
        :param type: 
            str, type of model to use

        :param tune:
            bool, whether to tune hyperparameters

        :param kwargs:
            dict, keyword arguments for model
        """
        super(RandomForest, self).__init__(
            type=type,
            **kwargs
        )

        if self.type == 'sklearn':
            self.model = RandomForestClassifier
        elif self.type == 'tf':
            self.model = tfdf.keras.RandomForestModel
        

class GradientBoosting(AbstractTreeModel):
    """ Gradient Boosting model """
    def __init__(
            self, 
            type: str = 'sklearn',
            **kwargs
    ):
        """ 
        :param type: 
            str, type of model to use

        :param tune:
            bool, whether to tune hyperparameters

        :param kwargs:
            dict, keyword arguments for model
        """
        super(GradientBoosting, self).__init__(
            type=type,
            **kwargs
        )

        if self.type == 'sklearn':
            self.model = GradientBoostingClassifier
        elif self.type == 'xgboost':
            self.model = XGBClassifier
        elif self.type == 'tf':
            self.model = tfdf.keras.GradientBoostedTreesModel

    