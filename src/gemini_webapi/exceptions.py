class AuthError(Exception):
    """
    Exception for authentication errors caused by invalid credentials/cookies.
    """

    pass


class APIError(Exception):
    """
    Exception for package-level errors which need to be fixed in the future development (e.g. validation errors).
    """

    pass


class ImageGenerationError(APIError):
    """
    Exception for generated image parsing errors.
    """

    pass


class GeminiError(Exception):
    """
    Exception for errors returned from Gemini server which are not handled by the package.
    """

    pass


class TimeoutError(GeminiError):
    """
    Exception for request timeouts.
    """

    pass


class UsageLimitExceeded(GeminiError):
    """
    Exception for model usage limit exceeded errors.
    """

    pass


class ModelInvalid(GeminiError):
    """
    Exception for invalid model header string errors.
    """

    pass


class TemporarilyBlocked(GeminiError):
    """
    Exception for 429 Too Many Requests when IP is temporarily blocked.
    """

    pass


class ServerError(GeminiError):
    """
    Exception for server-side generation failures reported via error status
    codes in batch execute responses (e.g., gRPC INTERNAL [13]).
    """

    pass


class WAATokenError(GeminiError):
    """WAA/BotGuard token harvesting failure. Non-retryable."""

    pass


class DeepThinkUnavailable(GeminiError):
    """Deep think request failed — server returned soft failure instead of content.
    Typically means the deep think quota is exhausted or the server couldn't process."""

    pass


class PayloadValidationError(GeminiError):
    """
    Raised when the client's request payload is silently rejected by the
    Gemini server for a specific model while a different model succeeds,
    indicating payload fingerprint drift.

    Subclass of GeminiError (NOT APIError) so ``@running`` does not retry it —
    the point is immediate surfacing of a bug that retries cannot fix.
    """

    pass
