import numpy as np
from .coresetmethod import CoresetMethod


class Oracle(CoresetMethod):
    """
    Coreset selection method that takes a .csv file with two columns: the first column contains the indices of the data,
    the second contains the score of the data. The method should select the points with the lowest / highest score
    (specified by the 'reverse' parameter). The file path should be passed as the 'file_path' parameter.
    """
    def __init__(self, dst_train, fraction=0.5, random_seed=None, file_path=None, reverse=False, **kwargs):
        """

        :param reverse:     If True, the points with the highest score are selected. If False, the points with the
                            lowest score are selected.
        """
        super().__init__(dst_train, fraction, random_seed)
        self.file_path = file_path
        self.reverse = reverse
        self.n_train = len(dst_train)

    def select(self, **kwargs):
        if self.file_path is None:
            raise ValueError('No file path specified for oracle method.')
        # load data and ignore header
        data = np.loadtxt(self.file_path, delimiter=',', skiprows=1)
        indices = data[:, 0].astype(np.int64)
        scores = data[:, 1]
        if self.reverse:
            indices = indices[np.argsort(-scores)]
        else:
            indices = indices[np.argsort(scores)]
        # all remaining data that we don't have scores for to the end
        indices = np.concatenate([indices, np.setdiff1d(np.arange(self.n_train), indices)])
        self.index = indices[:round(self.n_train * self.fraction)]
        return {"indices": self.index}