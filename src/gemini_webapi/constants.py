from enum import Enum, IntEnum, StrEnum


TEMPORARY_CHAT_FLAG_INDEX = 45


class Endpoint(StrEnum):
    GOOGLE = "https://www.google.com"
    INIT = "https://gemini.google.com/app"
    GENERATE = "https://gemini.google.com/_/BardChatUi/data/assistant.lamda.BardFrontendService/StreamGenerate"
    ROTATE_COOKIES = "https://accounts.google.com/RotateCookies"
    UPLOAD = "https://content-push.googleapis.com/upload"
    BATCH_EXEC = "https://gemini.google.com/_/BardChatUi/data/batchexecute"


class GRPC(StrEnum):
    """
    Google RPC ids used in Gemini API.
    """

    # Chat methods
    LIST_CHATS = "MaZiqc"
    READ_CHAT = "hNvQHb"
    DELETE_CHAT = "GzXR5e"

    # Gem methods
    LIST_GEMS = "CNgdBe"
    CREATE_GEM = "oMH3Zd"
    UPDATE_GEM = "kHv0Vd"
    DELETE_GEM = "UXcSJb"

    # Activity methods
    BARD_ACTIVITY = "ESY5D"


class Headers(Enum):
    GEMINI = {
        "Content-Type": "application/x-www-form-urlencoded;charset=utf-8",
        "Host": "gemini.google.com",
        "Origin": "https://gemini.google.com",
        "Referer": "https://gemini.google.com/",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36",
        "X-Same-Domain": "1",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Ch-Ua": '"Not:A-Brand";v="99", "Google Chrome";v="145", "Chromium";v="145"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"macOS"',
        "Sec-Ch-Ua-Full-Version-List": '"Not:A-Brand";v="99.0.0.0", "Google Chrome";v="145.0.7632.162", "Chromium";v="145.0.7632.162"',
        "Sec-Ch-Ua-Bitness": '"64"',
        "Sec-Ch-Ua-Model": '""',
        "Sec-Ch-Ua-Wow64": "?0",
        "Sec-Ch-Ua-Form-Factors": '"Desktop"',
        "Sec-Ch-Ua-Arch": '"arm"',
        "Sec-Ch-Ua-Full-Version": '"145.0.7632.162"',
        "Sec-Ch-Ua-Platform-Version": '"15.7.4"',
        "X-Browser-Channel": "stable",
        "X-Browser-Copyright": "Copyright 2026 Google LLC. All Rights reserved.",
        "X-Browser-Year": "2026",
    }
    ROTATE_COOKIES = {
        "Content-Type": "application/json",
    }
    UPLOAD = {"Push-ID": "feeds/mcudyrk2a4khkz"}


class Model(Enum):
    UNSPECIFIED = ("unspecified", {}, False)
    G_3_1_PRO = (
        "gemini-3.1-pro",
        {
            "x-goog-ext-525001261-jspb": '[1,null,null,null,"e6fa609c3fa255c0",null,null,0,[4],null,null,2]',
            "x-goog-ext-73010989-jspb": "[0]",
            "x-goog-ext-73010990-jspb": "[0]",
        },
        False,
    )
    G_3_0_FLASH = (
        "gemini-3.0-flash",
        {
            "x-goog-ext-525001261-jspb": '[1,null,null,null,"fbb127bbb056c959",null,null,0,[4],null,null,1]',
            "x-goog-ext-73010989-jspb": "[0]",
            "x-goog-ext-73010990-jspb": "[0]",
        },
        False,
    )
    G_3_0_FLASH_THINKING = (
        "gemini-3.0-flash-thinking",
        {
            "x-goog-ext-525001261-jspb": '[1,null,null,null,"5bf011840784117a",null,null,0,[4],null,null,1]',
            "x-goog-ext-73010989-jspb": "[0]",
            "x-goog-ext-73010990-jspb": "[0]",
        },
        False,
    )

    def __init__(self, name, header, advanced_only):
        self.model_name = name
        self.model_header = header
        self.advanced_only = advanced_only

    @classmethod
    def from_name(cls, name: str):
        # Legacy name mappings for backward compatibility
        legacy_names = {"gemini-3.0-pro": "gemini-3.1-pro"}
        resolved = legacy_names.get(name, name)

        for model in cls:
            if model.model_name == resolved:
                return model

        raise ValueError(
            f"Unknown model name: {name}. Available models: {', '.join([model.model_name for model in cls])}"
        )

    @classmethod
    def from_dict(cls, model_dict: dict):
        if "model_name" not in model_dict or "model_header" not in model_dict:
            raise ValueError(
                "When passing a custom model as a dictionary, 'model_name' and 'model_header' keys must be provided."
            )

        if not isinstance(model_dict["model_header"], dict):
            raise ValueError(
                "When passing a custom model as a dictionary, 'model_header' must be a dictionary containing valid header strings."
            )

        custom_model = cls.UNSPECIFIED
        custom_model.model_name = model_dict["model_name"]
        custom_model.model_header = model_dict["model_header"]
        return custom_model


class ErrorCode(IntEnum):
    """
    Known error codes returned from server.
    """

    TEMPORARY_ERROR_1013 = 1013  # Randomly raised when generating with certain models, but disappears soon after
    USAGE_LIMIT_EXCEEDED = 1037
    MODEL_INCONSISTENT = 1050
    MODEL_HEADER_INVALID = 1052
    IP_TEMPORARILY_BLOCKED = 1060
