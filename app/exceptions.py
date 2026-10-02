class NotFoundError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ConflictError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message
