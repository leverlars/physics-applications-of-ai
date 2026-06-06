"""Model factory helpers for jet-tagging classifiers."""

from sklearn.ensemble import HistGradientBoostingClassifier


def make_hist_gradient_boosting_classifier(
    *,
    max_iter: int,
    learning_rate: float = 0.08,
    l2_regularization: float = 0.02,
    random_state: int = 7,
) -> HistGradientBoostingClassifier:
    """Create the gradient-boosted tree classifier used across the notebook tasks."""
    return HistGradientBoostingClassifier(
        max_iter=max_iter,
        learning_rate=learning_rate,
        l2_regularization=l2_regularization,
        random_state=random_state,
    )
