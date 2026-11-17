"""Shared method and parameter-key validation for processing and evaluation."""


def validate_algorithm_params(method, params=None):
    """Return the normalized method and parameters, or raise ValueError."""
    if params is None:
        params = {}

    if not isinstance(params, dict):
        raise ValueError("params must be a dictionary")
    if not isinstance(method, str):
        raise ValueError("method must be gamma, he or clahe")
    method = method.lower()
    if method not in ("gamma", "he", "clahe"):
        raise ValueError(f"Thuật toán không hợp lệ: {method}")

    allowed_params = {
        "gamma": {"gamma"},
        "he": set(),
        "clahe": {"grid", "clip_factor"},
    }
    unknown = set(params) - allowed_params[method]
    if unknown:
        raise ValueError(f"Unexpected parameters for {method}: {list(unknown)}")
    return method, params
