class InferenceWorkerError(RuntimeError):
    """Base error for local inference worker failures."""


class InferenceWorkerUnavailable(InferenceWorkerError):
    pass


class InferenceWorkerTimeout(InferenceWorkerError):
    pass


class InferenceWorkerOOM(InferenceWorkerError):
    pass

