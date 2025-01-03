from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE

# Set a consistent random seed for reproducibility
SEED = 24

def no_sampling(X, Y):
    """
    Return the original dataset without any sampling.

    Args:
        X: Features.
        Y: Labels.
    
    Returns:
        X, Y: Unmodified input features and labels.
    """
    return X, Y

def up_sampling(X, Y):
    """
    Perform random oversampling to balance the dataset.

    Args:
        X: Features.
        Y: Labels.
        seed: Random seed for reproducibility (default: 24).
    
    Returns:
        X_resampled, Y_resampled: Oversampled features and labels.
    """
    ros = RandomOverSampler(random_state=SEED)
    X_resampled, Y_resampled = ros.fit_resample(X, Y)
    return X_resampled, Y_resampled

def down_sampling(X, Y):
    """
    Perform random undersampling to balance the dataset.

    Args:
        X: Features.
        Y: Labels.
        seed: Random seed for reproducibility (default: 24).
    
    Returns:
        X_resampled, Y_resampled: Undersampled features and labels.
    """
    rds = RandomUnderSampler(random_state=SEED)
    X_resampled, Y_resampled = rds.fit_resample(X, Y)
    return X_resampled, Y_resampled

def smote_sampling(X, Y):
    """
    Perform SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset.

    Args:
        X: Features.
        Y: Labels.
        seed: Random seed for reproducibility (default: 24).
    
    Returns:
        X_resampled, Y_resampled: SMOTE-processed features and labels.
    """
    sm = SMOTE(random_state=SEED)
    X_resampled, Y_resampled = sm.fit_resample(X, Y)
    return X_resampled, Y_resampled















