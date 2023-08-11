categoricals = [
    'SARS-Cov-2 exam result',
    'Influenza A',
    'Influenza B',
    'Parainfluenza 1',
    'CoronavirusNL63',
    'Rhinovirus/Enterovirus',
    'Coronavirus HKU1',
    'Parainfluenza 3',
    'Chlamydophila pneumoniae',
    'Respiratory Syncytial Virus',
    'Adenovirus',
    'Parainfluenza 4',
    'Coronavirus229E',
    'CoronavirusOC43',
    'Inf A H1N1 2009',
    'Bordetella pertussis',
    'Metapneumovirus',
    'Parainfluenza 2',
    'Influenza B, rapid test',
    'Influenza A, rapid test',
]

scale_cols = [
    # 'Patient addmited to intensive care unit (1=yes, 0=no)'
    ]

drop_cols = [
    "Patient ID",
    "Patient addmited to regular ward (1=yes, 0=no)",
    "Patient addmited to semi-intensive unit (1=yes, 0=no)",
    ]

params_tf = {
    'max_depth': range(5, 15, 2),
    # "use_hessian_gain": [True, False],
    "categorical_algorithm": ["CART", "RANDOM"],
    "num_trees": [50, 100, 150, 200],
    # "shrinkage": [0.02, 0.05, 0.10, 0.15],
    "num_candidate_attributes_ratio": [0.2, 0.5, 0.9, 1.0]
}

params_sklearn = {
    'max_depth': range(5, 15, 2),
    'n_estimators': (50, 100),
    'max_features': [.1, .2, .3, .4, .5, .6],
    'min_samples_leaf': [20, 30],
    'min_samples_split': [20, 30]
    }

params_xgb = {
    "learning_rate": (0.05, 0.10, 0.15),
    "max_depth": [3, 4, 5, 6, 8],
    "gamma": [0.1, 0.5, 1, .5, 2],
    "sumsamples": [0.25, 0.50, 0.75],
    "n_estimators": [50, 100, 150],
    "colsample_bytree":[0.25, 0.50]
} 

