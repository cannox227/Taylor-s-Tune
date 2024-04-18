import numpy as np

def cosine_similarity(x , y):
    '''
    To calculate how similar the profiles are in terms of their distribution across emotions, regardless of the intensity.

    Parameters:
    -----------
    x: numpy array of shape (n, d)
    y: numpy array of shape (n, d)

    Returns:
    --------
    cosine_similarity: numpy array of shape (n,)
    '''


    return np.sum(x * y, axis=1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1))


def euclidean_distance(x, y):
    '''
    To calculate how close the predicted emotions are to the actual emotions.
    '''
    squared_diff = np.square(x - y)
    # Sum the squared differences along the feature axis (D) and take the square root
    euclidean_dist = np.sqrt(np.sum(squared_diff, axis=1))
    return euclidean_dist