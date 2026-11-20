"""Small classifier heads trained on frozen backbone features.

Training on stored features makes each run take seconds on a CPU, so ensembles and many seeds are
cheap. An ensemble of logistic-regression heads with different seeds and bootstrap resamples
gives the deep-ensemble baseline.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class HeadEnsemble:
    def __init__(self, n_members: int = 5, C: float = 0.5, seed: int = 0, bootstrap: bool = True):
        self.n_members = n_members
        self.C = C
        self.seed = seed
        self.bootstrap = bootstrap
        self.scaler = StandardScaler()
        self.members: list[LogisticRegression] = []
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> HeadEnsemble:
        Xs = self.scaler.fit_transform(X)
        self.classes_ = np.unique(y)
        rng = np.random.default_rng(self.seed)
        self.members = []
        for m in range(self.n_members):
            if self.bootstrap:
                idx = rng.choice(len(y), size=len(y), replace=True)
                # make sure every class is present in the resample
                missing = np.setdiff1d(self.classes_, np.unique(y[idx]))
                if len(missing):
                    extra = np.concatenate([np.flatnonzero(y == c)[:1] for c in missing])
                    idx = np.concatenate([idx, extra])
            else:
                idx = np.arange(len(y))
            clf = LogisticRegression(C=self.C, max_iter=2000, random_state=self.seed + m)
            clf.fit(Xs[idx], y[idx])
            self.members.append(clf)
        return self

    def member_logits(self, X: np.ndarray) -> np.ndarray:
        """(m, n, k) decision-function logits for each member."""
        Xs = self.scaler.transform(X)
        out = []
        for clf in self.members:
            z = clf.decision_function(Xs)
            if z.ndim == 1:  # binary: sklearn returns one column
                z = np.stack([-z / 2, z / 2], axis=1)
            out.append(z)
        return np.stack(out)

    def member_probs(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        return np.stack([clf.predict_proba(Xs) for clf in self.members])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.member_probs(X).mean(axis=0)

    def single_member_proba(self, X: np.ndarray) -> np.ndarray:
        """Probabilities from member 0 only (the single-model baseline)."""
        return self.member_probs(X)[0]
