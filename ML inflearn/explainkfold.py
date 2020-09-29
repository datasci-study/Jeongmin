import numpy as np
from abc import * #추상클래스 사용하기 위한 모듈 설치
from ..utils import indexable, check_random_state, _safe_indexing

class BaseCrossValidator(metaclass=ABCMeta): #BaseCrossValidator 라는 추상클래스를 생성
    """Base class for all cross-validators
    Implementations must define `_iter_test_masks` or `_iter_test_indices`.
    """
    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            The target variable for supervised learning problems.
        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups) # indexable => 교차검증을 할수 있는 인덱스 배열을 만든다. 
        indices = np.arange(_num_samples(X))
        for test_index in self._iter_test_masks(X, y, groups):
            train_index = indices[np.logical_not(test_index)]
            test_index = indices[test_index]
            yield train_index, test_index
# Since subclasses must implement either _iter_test_masks or
    # _iter_test_indices, neither can be abstract.
    def _iter_test_masks(self, X=None, y=None, groups=None): #test set에 대응하는 boolean을 생성한다. 
        """Generates boolean masks corresponding to test sets.
        By default, delegates to _iter_test_indices(X, y, groups)
        """
        for test_index in self._iter_test_indices(X, y, groups):
            test_mask = np.zeros(_num_samples(X), dtype=np.bool)
            test_mask[test_index] = True
            yield test_mask

    def _iter_test_indices(self, X=None, y=None, groups=None): # test set에 대응하는 정수형 지수를 생성한다. 
        """Generates integer indices corresponding to test sets."""
        raise NotImplementedError # NotImplementedError는 파이썬 내장 오류로, 꼭 작성해야 하는 부분이 구현되지 않았을 경우 일부러 오류를 일으키기 위해 사용한다

    @abstractmethod # 하위 클래스가 추상 클래스를 상속받았다면 @abstractmethod가 붙은 추상 메서드를 모두 구현해야 합니다
    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator"""

    def __repr__(self):
        return _build_repr(self)





class _BaseKFold(BaseCrossValidator, metaclass=ABCMeta):
    
    """Base class for KFold, GroupKFold, and StratifiedKFold"""

    @abstractmethod
    @_deprecate_positional_args
    def __init__(self, n_splits, *, shuffle, random_state): # 인자를 입력했을 떄 오류값
        if not isinstance(n_splits, numbers.Integral):
            raise ValueError('The number of folds must be of Integral type. '
                             '%s of type %s was passed.'
                             % (n_splits, type(n_splits)))
        n_splits = int(n_splits)

        if n_splits <= 1: # 1이하의 n_splits값을 입력하엿을때
            raise ValueError(
                "k-fold cross-validation requires at least one"
                " train/test split by setting n_splits=2 or more,"
                " got n_splits={0}.".format(n_splits))

        if not isinstance(shuffle, bool): # boolean 이외의 다른 타입의 값이 입력되었을 때 오류
            raise TypeError("shuffle must be True or False;"
                            " got {0}".format(shuffle))

        if not shuffle and random_state is not None:  # None is the default
            # TODO 0.24: raise a ValueError instead of a warning
            warnings.warn(
                'Setting a random_state has no effect since shuffle is '
                'False. This will raise an error in 0.24. You should leave '
                'random_state to its default (None), or set shuffle=True.',
                FutureWarning
            )

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

class KFold(_BaseKFold):
    """K-Folds cross-validator
    Provides train/test indices to split data in train/test sets. Split
    dataset into k consecutive folds (without shuffling by default).
    Each fold is then used once as a validation while the k - 1 remaining
    folds form the training set.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
        .. versionchanged:: 0.22
            ``n_splits`` default value changed from 3 to 5.
    shuffle : bool, default=False
        Whether to shuffle the data before splitting into batches.
        Note that the samples within each split will not be shuffled.
    random_state : int or RandomState instance, default=None
        When `shuffle` is True, `random_state` affects the ordering of the
        indices, which controls the randomness of each fold. Otherwise, this
        parameter has no effect.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import KFold
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([1, 2, 3, 4])
    >>> kf = KFold(n_splits=2)
    >>> kf.get_n_splits(X)
    2
    >>> print(kf)
    KFold(n_splits=2, random_state=None, shuffle=False)
    >>> for train_index, test_index in kf.split(X):
    ...     print("TRAIN:", train_index, "TEST:", test_index)
    ...     X_train, X_test = X[train_index], X[test_index]
    ...     y_train, y_test = y[train_index], y[test_index]
    TRAIN: [2 3] TEST: [0 1]
    TRAIN: [0 1] TEST: [2 3]
    Notes
    -----
    The first ``n_samples % n_splits`` folds have size
    ``n_samples // n_splits + 1``, other folds have size
    ``n_samples // n_splits``, where ``n_samples`` is the number of samples.
    Randomized CV splitters may return different results for each call of
    split. You can make the results identical by setting `random_state`
    to an integer.
    See also
    --------
    StratifiedKFold
        Takes group information into account to avoid building folds with
        imbalanced class distributions (for binary or multiclass
        classification tasks).
    GroupKFold: K-fold iterator variant with non-overlapping groups.
    RepeatedKFold: Repeats K-Fold n times.
    """
    @_deprecate_positional_args
    def __init__(self, n_splits=5, *, shuffle=False,
                 random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle, #_baseKfold의 모듈을 상속받음
                         random_state=random_state)

    def _iter_test_indices(self, X, y=None, groups=None):
        n_samples = _num_samples(X)
        indices = np.arange(n_samples)
        if self.shuffle:
            check_random_state(self.random_state).shuffle(indices)

        n_splits = self.n_splits
        fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=np.int)
        fold_sizes[:n_samples % n_splits] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            yield indices[start:stop]
            current = stop
# _build_repr(self) => 구현
def _build_repr(self):
    # XXX This is copied from BaseEstimator's get_params
    cls = self.__class__ # class 내부에서 사용한다면, self.__class__로 그 클래스 변수를 참조한다. 
    init = getattr(cls.__init__, 'deprecated_original', cls.__init__) # cls의 생성자.
    # Ignore varargs, kw and default values and pop self
    init_signature = signature(init)
    # Consider the constructor parameters excluding 'self'
    if init is object.__init__:
        args = []
    else:
        args = sorted([p.name for p in init_signature.parameters.values()
                       if p.name != 'self' and p.kind != p.VAR_KEYWORD])
    class_name = self.__class__.__name__
    params = dict()
    for key in args:
        # We need deprecation warnings to always be on in order to
        # catch deprecated param values.
        # This is set in utils/__init__.py but it gets overwritten
        # when running under python3 somehow.
        warnings.simplefilter("always", FutureWarning)
        try:
            with warnings.catch_warnings(record=True) as w:
                value = getattr(self, key, None)
                if value is None and hasattr(self, 'cvargs'):
                    value = self.cvargs.get(key, None)
            if len(w) and w[0].category == FutureWarning:
                # if the parameter is deprecated, don't show it
                continue
        finally:
            warnings.filters.pop(0)
        params[key] = value

    return '%s(%s)' % (class_name, _pprint(params, offset=len(class_name)))