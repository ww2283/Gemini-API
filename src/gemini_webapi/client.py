import asyncio
import codecs
import io
import time
import random
import re
import uuid
from asyncio import Task
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

import orjson as json
from curl_cffi.requests import AsyncSession, Cookies, Response
from curl_cffi.requests.exceptions import ReadTimeout

from .components import ChatMixin, GemMixin
from .constants import (
    DEEP_THINK_FLAG_INDEX,
    DEEP_THINK_FLAG_VALUE,
    DEEP_THINK_PLACEHOLDER_MARKER,
    DEEP_THINK_SOFT_FAILURE_MARKERS,
    Endpoint,
    ErrorCode,
    GRPC,
    Headers,
    Model,
    TEMPORARY_CHAT_FLAG_INDEX,
)
from .exceptions import (
    APIError,
    AuthError,
    DeepThinkUnavailable,
    GeminiError,
    ModelInvalid,
    PayloadValidationError,
    ServerError,
    TemporarilyBlocked,
    TimeoutError,
    UsageLimitExceeded,
)
from .types import (
    Candidate,
    Gem,
    GeneratedImage,
    ModelOutput,
    RPCData,
    WebImage,
)
from .utils import (
    get_access_token,
    get_delta_by_fp_len,
    get_nested_value,
    logger,
    parse_file_name,
    parse_response_by_frame,
    rotate_1psidts,
    running,
    upload_file,
)


# Positions excluded from inner_req_list drift detection because they carry
# per-request dynamic content (message, metadata, tokens, UUIDs, gem IDs).
_DIFF_EXCLUDED_SLOTS: frozenset[int] = frozenset({
    0,   # message_content
    2,   # chat metadata
    3,   # WAA token
    4,   # botguard hash
    19,  # gem_id
    59,  # per-request UUID
    TEMPORARY_CHAT_FLAG_INDEX,
    DEEP_THINK_FLAG_INDEX,
})


def build_diagnostic_inner_req_list(
    model: "Model",
    prompt: str = "probe",
    chat_metadata: list | None = None,
) -> list:
    """Build the inner_req_list the library would send to StreamGenerate
    for the given model and prompt.

    Dynamic per-request slots (WAA token, botguard hash, UUID) are left
    as None. Used by the diag CLI to diff what the client WOULD send
    against what Chrome actually sends.
    """
    inner_req_list: list[Any] = [None] * 80
    inner_req_list[0] = [prompt, 0, None, None, None, None, 0]
    inner_req_list[1] = ["en"]
    inner_req_list[2] = (
        chat_metadata
        if chat_metadata is not None
        else ["", "", "", None, None, None, None, None, None, ""]
    )
    # slots 3 and 4 remain None (WAA token / botguard hash sentinels)
    inner_req_list[6] = [0]
    inner_req_list[7] = 1
    inner_req_list[10] = 1
    inner_req_list[11] = 0
    inner_req_list[17] = [[0]]
    inner_req_list[18] = 0
    inner_req_list[27] = 1
    inner_req_list[30] = [4]
    inner_req_list[41] = [1]
    inner_req_list[53] = 0
    # slot 59 stays None (per-request UUID)
    inner_req_list[61] = []
    inner_req_list[67] = 0
    inner_req_list[68] = 1
    jspb_str = model.model_header.get("x-goog-ext-525001261-jspb", "")
    if jspb_str:
        try:
            # Variant lives at jspb slot 14 in the current Chrome schema.
            # Slot 11 used to carry a duplicate variant marker but is now
            # null — do not fall back to it, a None value here would pin
            # the probe to a variant Google rejects.
            inner_req_list[79] = json.loads(jspb_str)[14]
        except Exception:
            pass
    return inner_req_list


class GeminiClient(ChatMixin, GemMixin):
    """
    Async client interface for gemini.google.com.

    `secure_1psid` must be provided unless the optional dependency `browser-cookie3` is installed, and
    you have logged in to google.com in your local browser.

    Parameters
    ----------
    secure_1psid: `str`, optional
        __Secure-1PSID cookie value.
    secure_1psidts: `str`, optional
        __Secure-1PSIDTS cookie value, some Google accounts don't require this value, provide only if it's in the cookie list.
    cookies: `dict[str, str]`, optional
        Full Google cookie dict for browser-parity (SID, HSID, SSID, etc.).
        Values are set on the `.google.com` domain. If both `cookies`
        and `secure_1psid` provide `__Secure-1PSID`, the explicit
        `secure_1psid` parameter takes precedence.
    proxy: `str`, optional
        Proxy URL.
    kwargs: `dict`, optional
        Additional arguments which will be passed to the http client.
        Refer to `curl_cffi.requests.AsyncSession` for more information.

    Raises
    ------
    `ValueError`
        If `browser-cookie3` is installed but cookies for google.com are not found in your local browser storage.
    """

    __slots__ = [
        "cookies",
        "proxy",
        "_running",
        "client",
        "access_token",
        "build_label",
        "session_id",
        "timeout",
        "auto_close",
        "close_delay",
        "close_task",
        "auto_refresh",
        "refresh_interval",
        "refresh_task",
        "verbose",
        "watchdog_timeout",
        "deep_think_poll_timeout",
        "read_chat_delays",
        "deep_think_soft_failure_markers",
        "_lock",
        "_reqid",
        "_gems",  # From GemMixin
        "waa_token_provider",
        "kwargs",
    ]

    def __init__(
        self,
        secure_1psid: str | None = None,
        secure_1psidts: str | None = None,
        proxy: str | None = None,
        cookies: dict[str, str] | None = None,
        waa_token_provider=None,
        **kwargs,
    ):
        super().__init__()
        self.cookies = Cookies()
        self.proxy = proxy
        self._running: bool = False
        self.client: AsyncSession | None = None
        self.access_token: str | None = None
        self.build_label: str | None = None
        self.session_id: str | None = None
        self._discovered_model_ids: dict[str, list[str]] = {}
        self._reference_inner_req_lists: dict[str, list] = {}
        self.timeout: float = 300
        self.auto_close: bool = False
        self.close_delay: float = 300
        self.close_task: Task | None = None
        self.auto_refresh: bool = True
        self.refresh_interval: float = 540
        self.refresh_task: Task | None = None
        self.verbose: bool = True
        self.watchdog_timeout: float = 30  # seconds before declaring a zombie stream
        self.deep_think_poll_timeout: float = 600  # max seconds to poll for deep think response
        self.read_chat_delays: list[float] = [30, 45, 60, 90]
        self.deep_think_soft_failure_markers: list[str] = list(
            DEEP_THINK_SOFT_FAILURE_MARKERS
        )
        self._lock = asyncio.Lock()
        self._reqid: int = random.randint(10000, 99999)
        self.kwargs = kwargs
        self.waa_token_provider = waa_token_provider

        # Forward caller-supplied cookies to .google.com domain.
        # secure_1psid/secure_1psidts below override matching keys.
        if cookies:
            for name, value in cookies.items():
                if value:
                    self.cookies.set(name, value, domain=".google.com")
                else:
                    logger.debug(f"Skipping cookie {name!r} with empty value")

        if secure_1psid:
            self.cookies.set("__Secure-1PSID", secure_1psid, domain=".google.com")
            if secure_1psidts:
                self.cookies.set(
                    "__Secure-1PSIDTS", secure_1psidts, domain=".google.com"
                )

    async def _get_waa_token(
        self, target_model_type: str | None = None
    ) -> tuple[str | None, str | None]:
        """Obtain a WAA/BotGuard token and paired hash from the configured provider.
        Also captures browser version for header matching and discovers
        current model IDs from the Gemini page.

        Returns:
            Tuple of (waa_token, botguard_hash). Either or both may be None.
        """
        if not self.waa_token_provider:
            return None, None
        try:
            botguard_hash = None
            if self.waa_token_provider is True:
                from .utils.waa_token import harvest_waa_token

                result = await harvest_waa_token(self.cookies)
                if isinstance(result, tuple):
                    if len(result) >= 6:
                        (
                            token,
                            browser_version,
                            model_ids,
                            botguard_hash,
                            reference_inner,
                            reference_headers,
                        ) = result[:6]
                        if model_ids:
                            self._discovered_model_ids = model_ids
                        if reference_inner is not None:
                            cache_key = target_model_type or "flash"
                            try:
                                self._reference_inner_req_lists[cache_key] = reference_inner
                            except AttributeError:
                                self._reference_inner_req_lists = {cache_key: reference_inner}
                        if isinstance(reference_headers, dict) and reference_headers:
                            try:
                                from .utils.jspb_patch import apply_autopatch
                                apply_autopatch(reference_headers)
                            except Exception as patch_err:
                                logger.debug(
                                    f"jspb autopatch skipped: {patch_err}"
                                )
                    elif len(result) == 5:
                        token, browser_version, model_ids, botguard_hash, reference_inner = result
                        if model_ids:
                            self._discovered_model_ids = model_ids
                        if reference_inner is not None:
                            cache_key = target_model_type or "flash"
                            try:
                                self._reference_inner_req_lists[cache_key] = reference_inner
                            except AttributeError:
                                self._reference_inner_req_lists = {cache_key: reference_inner}
                    elif len(result) == 4:
                        token, browser_version, model_ids, botguard_hash = result
                        if model_ids:
                            self._discovered_model_ids = model_ids
                    elif len(result) == 3:
                        token, browser_version, model_ids = result
                        if model_ids:
                            self._discovered_model_ids = model_ids
                    else:
                        token, browser_version = result
                    if browser_version:
                        self._chrome_version = browser_version
                else:
                    token = result
            elif callable(self.waa_token_provider):
                token = await self.waa_token_provider(self.cookies)
            else:
                return None, None
            if isinstance(token, str) and token.startswith("!"):
                return token, botguard_hash
            return None, None
        except Exception as e:
            logger.warning(f"WAA token harvesting failed: {e}")
            return None, None

    def _diff_inner_req_list(
        self, model_type: str, built_inner: list
    ) -> list[dict]:
        """Compare built_inner against the cached Chrome reference for model_type.

        Returns a list of drift entries describing slot-level differences.
        Slots in _DIFF_EXCLUDED_SLOTS are ignored. When no reference is cached
        for model_type, returns an empty list (degrades gracefully).
        """
        reference_cache = getattr(self, "_reference_inner_req_lists", {})
        if model_type not in reference_cache:
            return []
        reference = reference_cache[model_type]
        drift: list[dict] = []
        limit = min(len(built_inner), len(reference))
        for position in range(limit):
            if position in _DIFF_EXCLUDED_SLOTS:
                continue
            r = reference[position]
            c = built_inner[position]
            if r is None:
                continue
            if c is None:
                drift.append({
                    "position": position,
                    "client_value": None,
                    "chrome_value": r,
                    "kind": "missing_in_client",
                })
            elif r != c:
                drift.append({
                    "position": position,
                    "client_value": c,
                    "chrome_value": r,
                    "kind": "value_mismatch",
                })
        return drift

    async def diagnose_model(self, model_name: str = "gemini-3.1-pro") -> None:
        """Probe a model to detect payload fingerprint drift.

        Sends a trivial probe to ``model_name`` and to ``gemini-3.0-flash``.
        If the target model fails with a silent-stream ``APIError`` AND
        flash succeeds, raises :class:`PayloadValidationError` pointing
        at the diag CLI. Otherwise returns ``None``.

        Non-matching errors (quota, auth, outage) propagate unchanged so
        the probe cannot mask unrelated failures. This method is
        intentionally NOT wrapped with ``@running``: it is a diagnostic
        call, and the inner ``generate_content`` calls already carry
        their own retry policy. In real use, callers should expect that
        a failing probe will wait through ``generate_content``'s retries
        before surfacing.
        """
        target_error: APIError | None = None
        try:
            await self.generate_content(
                "reply one word: ok", model=model_name
            )
        except APIError as err:
            if "stream interrupted" in str(err).lower():
                target_error = err
            else:
                raise
        else:
            return None

        try:
            await self.generate_content(
                "reply one word: ok", model="gemini-3.0-flash"
            )
        except APIError:
            # Flash also broken — not a drift fingerprint; surface the
            # original target error unchanged.
            raise target_error

        raise PayloadValidationError(
            f"Target model {model_name!r} failed with the silent-stream "
            f"signature while gemini-3.0-flash succeeded. This is the "
            f"payload drift fingerprint. Run "
            f"'python -m gemini_webapi.diag --model pro' for a full "
            f"slot-by-slot diff."
        )

    def _build_chrome_headers(self) -> dict[str, str]:
        """Build sec-ch-ua headers matching the actual Chrome version."""
        version = getattr(self, "_chrome_version", None)
        if not version:
            return {}
        major = version.split(".")[0]
        return {
            "User-Agent": f"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{version} Safari/537.36",
            "Sec-Ch-Ua": f'"Chromium";v="{major}", "Not-A.Brand";v="24", "Google Chrome";v="{major}"',
            "Sec-Ch-Ua-Full-Version": f'"{version}"',
            "Sec-Ch-Ua-Full-Version-List": f'"Chromium";v="{version}", "Not-A.Brand";v="24.0.0.0", "Google Chrome";v="{version}"',
        }

    @property
    def discovered_model_ids(self) -> dict[str, list[str]]:
        """Model ID arrays discovered from the Gemini page.

        Keys are model types (``"pro"``, ``"flash"``, ``"thinking"``).
        Values are ordered lists of model IDs available for the
        authenticated account.  Populated after the first WAA harvest.
        """
        return self._discovered_model_ids

    # Map Model enum names to discovery keys
    _MODEL_TYPE_MAP: dict[str, str] = {
        "gemini-3.1-pro": "pro",
        "gemini-3.0-flash": "flash",
        "gemini-3.0-flash-thinking": "thinking",
    }

    def _resolve_model_header(
        self, model: "Model", target_variant: int | None = None,
    ) -> dict[str, str]:
        """Return model headers with the model ID resolved from discovery.

        Resolution strategy (requires ``_discovered_model_ids``):

        1. If *target_variant* is provided, select the model ID for that
           tier from the discovered array (last for highest, first for
           lowest) and rewrite both ID and variant in the header.
        2. Otherwise, if the header's current ID is absent from the
           discovered array (completely stale), replace with the last
           entry.
        3. If discovery data is unavailable, return headers unchanged.
        """
        header = dict(model.model_header)
        jspb_key = "x-goog-ext-525001261-jspb"
        jspb = header.get(jspb_key)
        if not jspb or not self._discovered_model_ids:
            return header

        model_type = self._MODEL_TYPE_MAP.get(model.model_name)
        if not model_type:
            return header

        discovered = self._discovered_model_ids.get(model_type)
        if not discovered:
            return header

        try:
            parsed = json.loads(jspb)
            current_id = parsed[4]
            current_variant = parsed[11]

            if target_variant is not None and target_variant != current_variant:
                # Caller requests a specific tier — pick from discovered array.
                # Heuristic: highest variant → last ID, lower → earlier IDs.
                if target_variant >= len(discovered):
                    new_id = discovered[-1]
                else:
                    new_id = discovered[max(0, target_variant - 1)]
                parsed[4] = new_id
                parsed[11] = target_variant
                # Ensure jspb has 15 elements with variant at position 14
                while len(parsed) < 15:
                    parsed.append(None)
                parsed[14] = target_variant
                header[jspb_key] = json.dumps(parsed).decode("utf-8")
                logger.info(
                    f"Resolved {model_type} model for variant {target_variant}: "
                    f"{current_id} -> {new_id}"
                )
            elif current_id not in discovered:
                # Completely stale — replace with last in array
                new_id = discovered[-1]
                new_variant = len(discovered)
                parsed[4] = new_id
                parsed[11] = new_variant
                header[jspb_key] = json.dumps(parsed).decode("utf-8")
                logger.info(
                    f"Auto-updated stale {model_type} model ID: "
                    f"{current_id} -> {new_id} (variant {new_variant})"
                )
        except Exception as e:
            logger.debug(f"Model ID resolution failed: {e}")

        return header

    async def init(
        self,
        timeout: float = 300,
        auto_close: bool = False,
        close_delay: float = 300,
        auto_refresh: bool = True,
        refresh_interval: float = 540,
        verbose: bool = True,
        watchdog_timeout: float = 30,  # seconds before declaring a zombie stream
        deep_think_poll_timeout: float = 600,  # max seconds to poll for deep think
        read_chat_delays: list[float] | None = None,
        deep_think_soft_failure_markers: list[str] | None = None,
    ) -> None:
        """
        Get SNlM0e value as access token. Without this token posting will fail with 400 bad request.

        Parameters
        ----------
        timeout: `float`, optional
            Request timeout of the client in seconds. Used to limit the max waiting time when sending a request.
        auto_close: `bool`, optional
            If `True`, the client will close connections and clear resource usage after a certain period
            of inactivity. Useful for always-on services.
        close_delay: `float`, optional
            Time to wait before auto-closing the client in seconds. Effective only if `auto_close` is `True`.
        auto_refresh: `bool`, optional
            If `True`, will schedule a task to automatically refresh cookies and access token in the background.
        refresh_interval: `float`, optional
            Time interval for background cookie and access token refresh in seconds.
            Effective only if `auto_refresh` is `True`.
        verbose: `bool`, optional
            If `True`, will print more infomation in logs.
        watchdog_timeout: `float`, optional
            Timeout in seconds for shadow retry watchdog. If no data receives from stream but connection is active,
            client will retry automatically after this duration.
        deep_think_poll_timeout: `float`, optional
            Max seconds to poll for deep think response. Deep think runs asynchronously on the server;
            the client polls READ_CHAT every 15s until the response is ready. Default 600s (10 min).
        read_chat_delays: `list[float]`, optional
            Recovery polling budget used when a stream breaks after `cid` was assigned or the
            server began processing. Each value is the number of seconds to sleep before the
            next `fetch_latest_chat_response(cid)` attempt. Total recovery window equals the
            sum. Default `[30, 45, 60, 90]` (~225s, 4 attempts). Raise for workloads where
            complex Pro requests with attachments legitimately take minutes to materialize.
        deep_think_soft_failure_markers: `list[str]`, optional
            Substrings that, when found in a deep think response, indicate the server returned
            a canned UI failure message (e.g. quota exhausted, capacity exhausted) instead of
            real reasoning output. Matching responses raise `DeepThinkUnavailable`. Default
            covers the two known English variants; extend here to cover future wording
            changes or localized variants without forking the library.
        """

        async with self._lock:
            if self._running:
                return

            try:
                self.verbose = verbose
                self.watchdog_timeout = watchdog_timeout
                self.deep_think_poll_timeout = deep_think_poll_timeout
                if read_chat_delays is not None:
                    self.read_chat_delays = list(read_chat_delays)
                if deep_think_soft_failure_markers is not None:
                    self.deep_think_soft_failure_markers = list(
                        deep_think_soft_failure_markers
                    )
                access_token, build_label, session_id, valid_cookies, session = (
                    await get_access_token(
                        base_cookies=self.cookies,
                        proxy=self.proxy,
                        verbose=self.verbose,
                        verify=self.kwargs.get("verify", True),
                    )
                )

                session.timeout = timeout
                self.client = session
                self.access_token = access_token
                self.cookies = valid_cookies
                self.build_label = build_label
                self.session_id = session_id
                self._running = True

                self.timeout = timeout
                self.auto_close = auto_close
                self.close_delay = close_delay
                if self.auto_close:
                    await self.reset_close_task()

                self.auto_refresh = auto_refresh
                self.refresh_interval = refresh_interval

                if self.refresh_task:
                    self.refresh_task.cancel()
                    self.refresh_task = None

                if self.auto_refresh:
                    self.refresh_task = asyncio.create_task(self.start_auto_refresh())

                if self.verbose:
                    logger.success("Gemini client initialized successfully.")
            except Exception:
                await self.close()
                raise

    async def close(self, delay: float = 0) -> None:
        """
        Close the client after a certain period of inactivity, or call manually to close immediately.

        Parameters
        ----------
        delay: `float`, optional
            Time to wait before closing the client in seconds.
        """

        if delay:
            await asyncio.sleep(delay)

        self._running = False

        if self.close_task:
            self.close_task.cancel()
            self.close_task = None

        if self.refresh_task:
            self.refresh_task.cancel()
            self.refresh_task = None

        if self.client:
            await self.client.close()
            self.client = None

    async def reset_connection(self) -> None:
        """
        Close and recreate the HTTP transport without re-fetching tokens.

        Use this instead of close() when the connection is broken (zombie stream,
        HTTP/2 errors) but authentication tokens are still valid. Avoids an
        unnecessary round-trip to gemini.google.com/app on every retry.
        """
        if self.client:
            await self.client.close()

        self.client = AsyncSession(
            impersonate="chrome",
            timeout=self.timeout,
            proxy=self.proxy,
            allow_redirects=True,
            headers=Headers.GEMINI.value,
            cookies=self.cookies,
        )

    async def reset_close_task(self) -> None:
        """
        Reset the timer for closing the client when a new request is made.
        """

        if self.close_task:
            self.close_task.cancel()
            self.close_task = None

        self.close_task = asyncio.create_task(self.close(self.close_delay))

    async def start_auto_refresh(self) -> None:
        """
        Start the background task to automatically refresh cookies.
        """
        if self.refresh_interval < 60:
            self.refresh_interval = 60

        while self._running:
            await asyncio.sleep(self.refresh_interval)

            if not self._running:
                break

            try:
                async with self._lock:
                    # Refresh all cookies in the background to keep the session alive.
                    new_1psidts, rotated_cookies = await rotate_1psidts(
                        self.client, verbose=self.verbose,
                    )
                    if rotated_cookies:
                        self.cookies.update(rotated_cookies)
                        if self.client:
                            self.client.cookies.update(rotated_cookies)

                    if new_1psidts:
                        if rotated_cookies:
                            logger.debug("Cookies refreshed (network update).")
                        else:
                            logger.debug("Cookies are up to date (cached).")
                    else:
                        logger.warning(
                            "Rotation response did not contain a new __Secure-1PSIDTS. "
                            "Session might expire soon if this persists."
                        )
            except asyncio.CancelledError:
                raise
            except AuthError:
                logger.warning(
                    "AuthError: Failed to refresh cookies. Retrying in next interval."
                )
            except Exception as e:
                logger.warning(f"Unexpected error while refreshing cookies: {e}")

    async def generate_content(
        self,
        prompt: str,
        files: list[str | Path | bytes | io.BytesIO] | None = None,
        model: Model | str | dict = Model.UNSPECIFIED,
        gem: Gem | str | None = None,
        chat: Optional["ChatSession"] = None,
        temporary: bool = False,
        **kwargs,
    ) -> ModelOutput:
        """
        Generates contents with prompt.

        Parameters
        ----------
        prompt: `str`
            Text prompt provided by user.
        files: `list[str | Path | bytes | io.BytesIO]`, optional
            List of file paths or byte streams to be attached.
        model: `Model | str | dict`, optional
            Specify the model to use for generation.
            Pass either a `gemini_webapi.constants.Model` enum or a model name string to use predefined models.
            Pass a dictionary to use custom model header strings ("model_name" and "model_header" keys must be provided).
        gem: `Gem | str`, optional
            Specify a gem to use as system prompt for the chat session.
            Pass either a `gemini_webapi.types.Gem` object or a gem id string.
        chat: `ChatSession`, optional
            Chat data to retrieve conversation history.
            If None, will automatically generate a new chat id when sending post request.
        temporary: `bool`, optional
            If set to `True`, the ongoing conversation will not show up in Gemini history.
        kwargs: `dict`, optional
            Additional arguments which will be passed to the post request.
            Refer to `curl_cffi.requests.AsyncSession` for more information.

        Returns
        -------
        :class:`ModelOutput`
            Output data from gemini.google.com.

        Raises
        ------
        `AssertionError`
            If prompt is empty.
        `gemini_webapi.TimeoutError`
            If request timed out.
        `gemini_webapi.GeminiError`
            If no reply candidate found in response.
        `gemini_webapi.APIError`
            - If request failed with status code other than 200.
            - If response structure is invalid and failed to parse.
        """

        if self.auto_close:
            await self.reset_close_task()

        if not (isinstance(chat, ChatSession) and chat.cid):
            self._reqid = random.randint(10000, 99999)

        file_data = None
        if files:
            await self._batch_execute(
                [
                    RPCData(
                        rpcid=GRPC.BARD_ACTIVITY,
                        payload='[[["bard_activity_enabled"]]]',
                    )
                ]
            )

            uploaded_urls = await asyncio.gather(
                *(upload_file(file, self.proxy) for file in files)
            )
            file_data = [
                [[url], parse_file_name(file)]
                for url, file in zip(uploaded_urls, files)
            ]

        try:
            await self._batch_execute(
                [
                    RPCData(
                        rpcid=GRPC.BARD_ACTIVITY,
                        payload='[[["bard_activity_enabled"]]]',
                    )
                ]
            )

            session_state = {
                "last_texts": {},
                "last_thoughts": {},
                "last_progress_time": time.time(),
            }
            output = None
            async for output in self._generate(
                prompt=prompt,
                req_file_data=file_data,
                model=model,
                gem=gem,
                chat=chat,
                temporary=temporary,
                session_state=session_state,
                **kwargs,
            ):
                pass

            if output is None:
                raise GeminiError(
                    "Failed to generate contents. No output data found in response."
                )

            # Detect deep think soft failure (quota exhausted, server capacity, etc.)
            if (
                kwargs.get("deep_think")
                and output.text
                and any(
                    marker in output.text
                    for marker in self.deep_think_soft_failure_markers
                )
            ):
                raise DeepThinkUnavailable(output.text.strip())

            if isinstance(chat, ChatSession):
                output.metadata = chat.metadata
                chat.last_output = output

            return output

        finally:
            if files:
                for file in files:
                    if isinstance(file, io.BytesIO):
                        file.close()

    async def generate_content_stream(
        self,
        prompt: str,
        files: list[str | Path | bytes | io.BytesIO] | None = None,
        model: Model | str | dict = Model.UNSPECIFIED,
        gem: Gem | str | None = None,
        chat: Optional["ChatSession"] = None,
        temporary: bool = False,
        **kwargs,
    ) -> AsyncGenerator[ModelOutput, None]:
        """
        Generates contents with prompt in streaming mode.

        This method sends a request to Gemini and yields partial responses as they arrive.
        It automatically calculates the text delta (new characters) to provide a smooth
        streaming experience. It also continuously updates chat metadata and candidate IDs.

        Parameters
        ----------
        prompt: `str`
            Text prompt provided by user.
        files: `list[str | Path | bytes | io.BytesIO]`, optional
            List of file paths or byte streams to be attached.
        model: `Model | str | dict`, optional
            Specify the model to use for generation.
        gem: `Gem | str`, optional
            Specify a gem to use as system prompt for the chat session.
        chat: `ChatSession`, optional
            Chat data to retrieve conversation history.
        temporary: `bool`, optional
            If set to `True`, the ongoing conversation will not show up in Gemini history.
        kwargs: `dict`, optional
            Additional arguments passed to `curl_cffi.requests.AsyncSession.stream`.

        Yields
        ------
        :class:`ModelOutput`
            Partial output data. The `text_delta` attribute contains only the NEW characters
            received since the last yield.

        Raises
        ------
        `gemini_webapi.APIError`
            If the request fails or response structure is invalid.
        `gemini_webapi.TimeoutError`
            If the stream request times out.
        """

        if self.auto_close:
            await self.reset_close_task()

        if not (isinstance(chat, ChatSession) and chat.cid):
            self._reqid = random.randint(10000, 99999)

        file_data = None
        if files:
            await self._batch_execute(
                [
                    RPCData(
                        rpcid=GRPC.BARD_ACTIVITY,
                        payload='[[["bard_activity_enabled"]]]',
                    )
                ]
            )

            uploaded_urls = await asyncio.gather(
                *(upload_file(file, self.proxy) for file in files)
            )
            file_data = [
                [[url], parse_file_name(file)]
                for url, file in zip(uploaded_urls, files)
            ]

        try:
            await self._batch_execute(
                [
                    RPCData(
                        rpcid=GRPC.BARD_ACTIVITY,
                        payload='[[["bard_activity_enabled"]]]',
                    )
                ]
            )

            session_state = {
                "last_texts": {},
                "last_thoughts": {},
                "last_progress_time": time.time(),
            }
            output = None
            async for output in self._generate(
                prompt=prompt,
                req_file_data=file_data,
                model=model,
                gem=gem,
                chat=chat,
                temporary=temporary,
                session_state=session_state,
                **kwargs,
            ):
                yield output

            if output and isinstance(chat, ChatSession):
                output.metadata = chat.metadata
                chat.last_output = output

        finally:
            if files:
                for file in files:
                    if isinstance(file, io.BytesIO):
                        file.close()

    @running(retry=5)
    async def _generate(
        self,
        prompt: str,
        req_file_data: list[Any] | None = None,
        model: Model | str | dict = Model.UNSPECIFIED,
        gem: Gem | str | None = None,
        chat: Optional["ChatSession"] = None,
        temporary: bool = False,
        deep_think: bool = False,
        session_state: dict[str, Any] | None = None,
        **kwargs,
    ) -> AsyncGenerator[ModelOutput, None]:
        """
        Internal method which actually sends content generation requests.
        """

        assert prompt, "Prompt cannot be empty."

        if isinstance(model, str):
            model = Model.from_name(model)
        elif isinstance(model, dict):
            model = Model.from_dict(model)
        elif not isinstance(model, Model):
            raise TypeError(
                f"'model' must be a `gemini_webapi.constants.Model` instance, "
                f"string, or dictionary; got `{type(model).__name__}`"
            )

        _reqid = self._reqid
        self._reqid += 100000

        gem_id = gem.id if isinstance(gem, Gem) else gem

        try:
            message_content = [
                prompt,
                0,
                None,
                req_file_data,
                None,
                None,
                0,
            ]

            params: dict[str, Any] = {"_reqid": _reqid, "rt": "c"}
            if self.build_label:
                params["bl"] = self.build_label
            if self.session_id:
                params["f.sid"] = self.session_id

            inner_req_list: list[Any] = [None] * 80
            inner_req_list[0] = message_content

            # Clear stale rid/rcid that leaked from a failed stream attempt.
            # A partially-set rid with empty cid is an inconsistent state that
            # causes the server to reject retries.
            if isinstance(chat, ChatSession) and not chat.cid and chat.rid:
                chat.rid = ""
                chat.rcid = ""

            inner_req_list[2] = (
                chat.metadata
                if chat
                else ["", "", "", None, None, None, None, None, None, ""]
            )
            inner_req_list[7] = 1  # Enable Snapshot Streaming
            if gem_id:
                inner_req_list[19] = gem_id
            if temporary:
                inner_req_list[TEMPORARY_CHAT_FLAG_INDEX] = 1
            if deep_think:
                inner_req_list[DEEP_THINK_FLAG_INDEX] = DEEP_THINK_FLAG_VALUE

            # Browser-parity: fixed slots observed in Chrome network traces
            inner_req_list[1] = ["en"]
            inner_req_list[6] = [0]
            inner_req_list[10] = 1
            inner_req_list[11] = 0
            inner_req_list[17] = [[0]]
            inner_req_list[18] = 0
            inner_req_list[27] = 1
            inner_req_list[30] = [4]
            inner_req_list[41] = [1]
            inner_req_list[53] = 0
            inner_req_list[61] = []
            inner_req_list[67] = 0
            inner_req_list[68] = 1
            # Slot 79: model variant extracted from jspb header position 14.
            # Chrome's current schema keeps variant only at slot 14; slot 11
            # is null. See utils.jspb_patch for the canonical template.
            jspb_str = model.model_header.get("x-goog-ext-525001261-jspb", "")
            if jspb_str:
                try:
                    inner_req_list[79] = json.loads(jspb_str)[14]
                except Exception:
                    pass

            # WAA/BotGuard attestation token + paired hash for extended stream lifetime.
            waa_token, botguard_hash = await self._get_waa_token()
            if waa_token:
                inner_req_list[3] = waa_token
            if botguard_hash:
                inner_req_list[4] = botguard_hash

            # Pop library-internal kwargs before they leak to curl_cffi
            target_variant = kwargs.pop("target_variant", None)

            # Per-request UUID shared between inner_req_list[59] and header
            uuid_val = str(uuid.uuid4())
            inner_req_list[59] = uuid_val

            request_headers = {
                **self._resolve_model_header(model, target_variant=target_variant),
                **self._build_chrome_headers(),
                "x-goog-ext-525005358-jspb": f'["{uuid_val}",1]',
            }

            request_data = {
                "at": self.access_token,
                "f.req": json.dumps(
                    [
                        None,
                        json.dumps(inner_req_list).decode("utf-8"),
                    ]
                ).decode("utf-8"),
            }

            if session_state is not None:
                if "original_cid" not in session_state:
                    session_state["original_cid"] = chat.cid if chat else None
                if "original_rcid" not in session_state:
                    session_state["original_rcid"] = (
                        chat.rcid if isinstance(chat, ChatSession) else None
                    )
                    # Cross-session continuation: clear stale rid/rcid on the
                    # first attempt so the server determines the append point
                    # from cid alone.  Stale rid/rcid causes the server to
                    # queue-and-drop the request without processing.
                    # Only for fresh ChatSessions (last_output is None) that
                    # carry saved metadata from a previous session.
                    if (
                        isinstance(chat, ChatSession)
                        and chat.cid
                        and chat.rid
                        and chat.last_output is None
                    ):
                        logger.debug(
                            f"Clearing stale rid/rcid for cross-session continuation "
                            f"(cid={chat.cid!r}, rid={chat.rid!r}, rcid={chat.rcid!r})"
                        )
                        chat.rid = ""
                        chat.rcid = ""
                        # Rebuild metadata in request payload with cleared rid/rcid
                        inner_req_list[2] = chat.metadata
                # Sticky flag: once True, persists across retries so subsequent
                # attempts know the server started processing our prompt.
                if "had_response_data" not in session_state:
                    session_state["had_response_data"] = False

                if (
                    chat
                    and chat.cid
                    and (
                        session_state.get("original_cid") in ("", None)
                        or session_state.get("had_response_data")
                    )
                ):
                    # Recovery triggers in two cases:
                    # 1. New conversation: cid was empty, assigned mid-stream
                    # 2. Continuation: server started processing (had_response_data)
                    #    before stream broke — retrying would duplicate the turn
                    # Poll read_chat with exponential backoff. Google's Pro backend
                    # needs ~50-60s to persist responses after a stream break;
                    # complex Pro + attachments can take much longer. Budget is
                    # configurable via GeminiClient.read_chat_delays / init().
                    read_chat_delays = self.read_chat_delays
                    all_stale = True  # Track if every attempt returned stale
                    server_confirmed_failure = False  # ServerError [5] = drift fingerprint
                    for attempt, delay in enumerate(read_chat_delays, 1):
                        logger.warning(
                            f"Stream failed after Gemini assigned cid={chat.cid!r}. "
                            f"READ_CHAT attempt {attempt}/{len(read_chat_delays)}: "
                            f"waiting {delay}s for server to persist..."
                        )
                        await asyncio.sleep(delay)
                        try:
                            recovered = await self.fetch_latest_chat_response(chat.cid)
                            if recovered:
                                # Guard: for continuations, check if read_chat returned a stale
                                # response from a previous turn (same rcid as what we started with)
                                original_cid = session_state.get("original_cid")
                                if (
                                    original_cid
                                    and recovered.rcid
                                    and recovered.rcid
                                    == session_state.get("original_rcid")
                                ):
                                    logger.debug(
                                        f"READ_CHAT attempt {attempt} returned stale response "
                                        f"(rcid={recovered.rcid!r} matches original). Continuing..."
                                    )
                                    continue  # keep polling — new response not persisted yet

                                logger.info(
                                    f"Successfully recovered response via READ_CHAT "
                                    f"(attempt {attempt}/{len(read_chat_delays)})."
                                )
                                if isinstance(chat, ChatSession):
                                    chat.metadata = recovered.metadata
                                yield recovered
                                return
                            all_stale = False
                            logger.debug(f"READ_CHAT attempt {attempt} returned None")
                        except ServerError:
                            all_stale = False
                            server_confirmed_failure = True
                            logger.warning(
                                f"READ_CHAT attempt {attempt}: server confirmed "
                                f"generation failure. Skipping remaining attempts."
                            )
                            break
                        except Exception as e:
                            all_stale = False
                            logger.warning(
                                f"READ_CHAT attempt {attempt} failed for cid={chat.cid!r}: "
                                f"{type(e).__name__}: {e}"
                            )

                    if all_stale:
                        # Every attempt returned the previous turn's response —
                        # the server never processed our prompt, safe to retry.
                        # Drop rid/rcid so the retry doesn't send stale
                        # continuation metadata — let the server determine
                        # the append point from cid alone.
                        if isinstance(chat, ChatSession):
                            chat.rid = ""
                            chat.rcid = ""
                        session_state["had_response_data"] = False
                        raise APIError(
                            f"Stream failed for cid={chat.cid!r}. "
                            f"All {len(read_chat_delays)} READ_CHAT attempts returned stale "
                            f"response (rcid unchanged). Retrying stream."
                        )
                    if server_confirmed_failure:
                        # Server returned batch-execute status [5] (gRPC INTERNAL)
                        # on read_chat — it accepted the request but rejected the
                        # generation. The known cause is x-goog-ext-525001261-jspb
                        # header drift: non-null values in slots the current
                        # Chrome sends as null trigger a server-side guard that
                        # fires only on long streams. Surface as a drift error
                        # so @running does not retry with the same broken header.
                        raw_model_name = getattr(model, "model_name", model)
                        model_name_str = (
                            raw_model_name
                            if isinstance(raw_model_name, str)
                            else str(raw_model_name)
                        )
                        diag_alias = self._MODEL_TYPE_MAP.get(model_name_str, "pro")
                        raise PayloadValidationError(
                            f"Stream failed for cid={chat.cid!r} on model "
                            f"{model_name_str!r}; server confirmed generation "
                            "failure via batch-execute status [5]. This is "
                            "the payload-drift fingerprint (typically the "
                            "x-goog-ext-525001261-jspb header). Run "
                            f"'python -m gemini_webapi.diag --model {diag_alias} "
                            "--cdp-url http://localhost:9222' to diff against "
                            "a live Chrome capture."
                        )
                    # Some attempts returned None/errors — turn may exist
                    # server-side. GeminiError prevents @running from retrying.
                    raise GeminiError(
                        f"Stream failed after Gemini assigned cid={chat.cid!r}. "
                        f"Recovery via READ_CHAT returned no data after "
                        f"{len(read_chat_delays)} attempts (~{sum(read_chat_delays)}s). "
                        "Retrying would create a duplicate conversation thread."
                    )

            async with self.client.stream(
                "POST",
                Endpoint.GENERATE,
                params=params,
                headers=request_headers,
                data=request_data,
                **kwargs,
            ) as response:
                if response.status_code != 200:
                    await self.close()
                    raise APIError(
                        f"Failed to generate contents. Status: {response.status_code}"
                    )

                if self.client:
                    self.cookies.update(self.client.cookies)

                buffer = ""
                decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

                # Track last seen content. session_state allows persistence across retries.
                if session_state is None:
                    session_state = {
                        "last_texts": {},
                        "last_thoughts": {},
                        "last_progress_time": time.time(),
                    }

                last_texts: dict[str, str] = session_state["last_texts"]
                last_thoughts: dict[str, str] = session_state["last_thoughts"]
                # Reset watchdog timer per stream attempt; stale values from
                # previous retries would cause immediate false stall detection.
                last_progress_time = time.time()
                session_state["last_progress_time"] = last_progress_time

                is_thinking = False
                is_queueing = False
                has_candidates = False
                is_completed = False
                is_final_chunk = False
                is_deep_think_pending = False

                async def _process_parts(
                    parts: list[Any],
                ) -> AsyncGenerator[ModelOutput, None]:
                    nonlocal is_thinking, is_queueing, has_candidates, is_completed, is_final_chunk, is_deep_think_pending
                    for part in parts:
                        # 1. Check for fatal error codes
                        error_code = get_nested_value(part, [5, 2, 0, 1, 0])
                        if error_code:
                            await self.close()
                            match error_code:
                                case ErrorCode.USAGE_LIMIT_EXCEEDED:
                                    raise UsageLimitExceeded(
                                        f"Usage limit exceeded for model '{model.model_name}'. Please wait a few minutes, "
                                        "switch to a different model (e.g., Gemini Flash), or check your account limits on gemini.google.com."
                                    )
                                case ErrorCode.MODEL_INCONSISTENT:
                                    raise ModelInvalid(
                                        "The specified model is inconsistent with the conversation history. "
                                        "Please ensure you are using the same 'model' parameter throughout the entire ChatSession."
                                    )
                                case ErrorCode.MODEL_HEADER_INVALID:
                                    raise ModelInvalid(
                                        f"The model '{model.model_name}' is currently unavailable or the request structure is outdated. "
                                        "Please update 'gemini_webapi' to the latest version or report this on GitHub if the problem persists."
                                    )
                                case ErrorCode.IP_TEMPORARILY_BLOCKED:
                                    raise TemporarilyBlocked(
                                        "Your IP address has been temporarily flagged or blocked by Google. "
                                        "Please try using a proxy, a different network, or wait for a while before retrying."
                                    )
                                case ErrorCode.TEMPORARY_ERROR_1013:
                                    raise APIError(
                                        "Gemini encountered a temporary error (1013). Retrying..."
                                    )
                                case _:
                                    raise APIError(
                                        f"Failed to generate contents (stream). Unknown API error code: {error_code}. "
                                        "This might be a temporary Google service issue."
                                    )

                        # 2. Detect if model is busy analyzing data (Thinking state)
                        if "data_analysis_tool" in str(part):
                            is_thinking = True
                            if not has_candidates:
                                logger.debug("Model is active (thinking/analyzing)...")

                        # 3. Check for queueing status
                        status = get_nested_value(part, [5])
                        if isinstance(status, list) and status:
                            is_queueing = True
                            if not has_candidates:
                                logger.debug(
                                    "Model is in a waiting state (queueing)..."
                                )

                        inner_json_str = get_nested_value(part, [2])
                        if inner_json_str:
                            try:
                                part_json = json.loads(inner_json_str)
                                m_data = get_nested_value(part_json, [1])
                                if m_data and isinstance(chat, ChatSession):
                                    chat.metadata = m_data
                                context_str = get_nested_value(part_json, [25])
                                if isinstance(context_str, str):
                                    is_completed = True
                                    is_thinking = False
                                    is_queueing = False
                                    if isinstance(chat, ChatSession):
                                        chat.metadata = [None] * 9 + [context_str]

                                candidates_list = get_nested_value(part_json, [4], [])
                                if candidates_list:
                                    output_candidates = []
                                    for i, candidate_data in enumerate(candidates_list):
                                        rcid = get_nested_value(candidate_data, [0])
                                        if not rcid:
                                            continue
                                        if isinstance(chat, ChatSession):
                                            chat.rcid = rcid

                                        # Text output and thoughts
                                        text = get_nested_value(
                                            candidate_data, [1, 0], ""
                                        )
                                        if re.match(
                                            r"^http://googleusercontent\.com/card_content/\d+",
                                            text,
                                        ):
                                            text = (
                                                get_nested_value(
                                                    candidate_data, [22, 0]
                                                )
                                                or text
                                            )

                                        # Detect deep think placeholder before cleanup
                                        if DEEP_THINK_PLACEHOLDER_MARKER in text:
                                            is_deep_think_pending = True

                                        # Cleanup googleusercontent artifacts
                                        text = re.sub(
                                            r"http://googleusercontent\.com/\w+/\d+\n*",
                                            "",
                                            text,
                                        )

                                        thoughts = (
                                            get_nested_value(candidate_data, [37, 0, 0])
                                            or ""
                                        )

                                        # Image handling
                                        web_images = []
                                        web_img_list = get_nested_value(
                                            candidate_data, [12, 1], []
                                        )
                                        for web_img_data in (
                                            web_img_list
                                            if isinstance(web_img_list, list)
                                            else []
                                        ):
                                            url = get_nested_value(
                                                web_img_data, [0, 0, 0]
                                            )
                                            if url:
                                                web_images.append(
                                                    WebImage(
                                                        url=url,
                                                        title=get_nested_value(
                                                            web_img_data, [7, 0], ""
                                                        ),
                                                        alt=get_nested_value(
                                                            web_img_data, [0, 4], ""
                                                        ),
                                                        proxy=self.proxy,
                                                    )
                                                )

                                        generated_images = []
                                        gen_img_list = get_nested_value(
                                            candidate_data, [12, 7, 0], []
                                        )
                                        for gen_img_data in (
                                            gen_img_list
                                            if isinstance(gen_img_list, list)
                                            else []
                                        ):
                                            url = get_nested_value(
                                                gen_img_data, [0, 3, 3]
                                            )
                                            if url:
                                                img_num = get_nested_value(
                                                    gen_img_data, [3, 6]
                                                )
                                                generated_images.append(
                                                    GeneratedImage(
                                                        url=url,
                                                        title=(
                                                            f"[Generated Image {img_num}]"
                                                            if img_num
                                                            else "[Generated Image]"
                                                        ),
                                                        alt=get_nested_value(
                                                            gen_img_data, [3, 5, 0], ""
                                                        ),
                                                        proxy=self.proxy,
                                                        cookies=self.cookies,
                                                    )
                                                )

                                        # Determine if this frame represents the final state of the message
                                        is_final_chunk = (
                                            isinstance(
                                                get_nested_value(candidate_data, [2]),
                                                list,
                                            )
                                            or get_nested_value(
                                                candidate_data, [8, 0], 1
                                            )
                                            == 2
                                        )

                                        last_sent_text = last_texts.get(
                                            rcid
                                        ) or last_texts.get(f"idx_{i}", "")
                                        text_delta, new_full_text = get_delta_by_fp_len(
                                            text,
                                            last_sent_text,
                                            is_final=is_final_chunk,
                                        )
                                        last_sent_thought = last_thoughts.get(
                                            rcid
                                        ) or last_thoughts.get(f"idx_{i}", "")
                                        if thoughts:
                                            thoughts_delta, new_full_thought = (
                                                get_delta_by_fp_len(
                                                    thoughts,
                                                    last_sent_thought,
                                                    is_final=is_final_chunk,
                                                )
                                            )
                                        else:
                                            thoughts_delta = ""
                                            new_full_thought = ""

                                        if (
                                            text_delta
                                            or thoughts_delta
                                            or web_images
                                            or generated_images
                                        ):
                                            has_candidates = True

                                        # Update state with the provider's cleaned state to handle drift
                                        last_texts[rcid] = last_texts[f"idx_{i}"] = (
                                            new_full_text
                                        )

                                        last_thoughts[rcid] = last_thoughts[
                                            f"idx_{i}"
                                        ] = new_full_thought

                                        output_candidates.append(
                                            Candidate(
                                                rcid=rcid,
                                                text=text,
                                                text_delta=text_delta,
                                                thoughts=thoughts or None,
                                                thoughts_delta=thoughts_delta,
                                                web_images=web_images,
                                                generated_images=generated_images,
                                            )
                                        )

                                    if output_candidates:
                                        is_thinking = False
                                        is_queueing = False
                                        if session_state is not None:
                                            session_state["had_response_data"] = True
                                        yield ModelOutput(
                                            metadata=get_nested_value(
                                                part_json, [1], []
                                            ),
                                            candidates=output_candidates,
                                        )
                            except json.JSONDecodeError:
                                continue

                async for chunk in response.aiter_content():
                    buffer += decoder.decode(chunk, final=False)
                    if buffer.startswith(")]}'"):
                        buffer = buffer[4:].lstrip()
                    parsed_parts, buffer = parse_response_by_frame(buffer)

                    got_update = False
                    async for out in _process_parts(parsed_parts):
                        yield out
                        got_update = True

                    # Reset watchdog when data chunks actually arrive.
                    # Previously, is_thinking/is_queueing kept resetting the
                    # watchdog even when no new data was received, masking
                    # zombie streams during the thinking phase.
                    if parsed_parts or got_update:
                        last_progress_time = time.time()
                        session_state["last_progress_time"] = last_progress_time
                    else:
                        stall_threshold = self.watchdog_timeout
                        elapsed = time.time() - last_progress_time
                        if elapsed > stall_threshold:
                            logger.warning(
                                f"Response stalled (no data for {elapsed:.0f}s, "
                                f"thinking={is_thinking}, queueing={is_queueing}). Retrying..."
                            )
                            await self.reset_connection()
                            if is_queueing and not has_candidates:
                                raise APIError(
                                    "Gemini server is overloaded (request queued but never started processing). "
                                    "Try again in a few minutes or use a different model."
                                )
                            raise APIError("Response stalled (zombie stream).")

                # Final flush
                buffer += decoder.decode(b"", final=True)
                if buffer:
                    parsed_parts, _ = parse_response_by_frame(buffer)
                    async for out in _process_parts(parsed_parts):
                        yield out

                # Deep think: stream returns a placeholder immediately.
                # The real response arrives asynchronously; poll READ_CHAT.
                if is_deep_think_pending and chat and chat.cid:
                    poll_timeout = self.deep_think_poll_timeout
                    logger.info(
                        f"Deep think placeholder detected for cid={chat.cid!r}. "
                        f"Polling READ_CHAT (budget={poll_timeout}s)..."
                    )
                    poll_interval = 15  # seconds between polls
                    elapsed = 0.0
                    attempt = 0
                    while elapsed < poll_timeout:
                        await asyncio.sleep(poll_interval)
                        elapsed += poll_interval
                        attempt += 1
                        try:
                            recovered = await self.fetch_latest_chat_response(chat.cid)
                            if recovered and recovered.text:
                                if DEEP_THINK_PLACEHOLDER_MARKER in recovered.text:
                                    logger.debug(
                                        f"Deep think poll {attempt} "
                                        f"({elapsed:.0f}/{poll_timeout}s): "
                                        "still processing..."
                                    )
                                    continue
                                logger.info(
                                    f"Deep think response ready after "
                                    f"{elapsed:.0f}s (poll {attempt})."
                                )
                                if isinstance(chat, ChatSession):
                                    chat.metadata = recovered.metadata
                                yield recovered
                                return
                        except Exception as e:
                            logger.warning(
                                f"Deep think poll {attempt} failed: "
                                f"{type(e).__name__}: {e}"
                            )
                    raise GeminiError(
                        f"Deep think response not ready after "
                        f"{poll_timeout}s of polling for cid={chat.cid!r}. "
                        f"The response may still be processing server-side. "
                        f"Try client.read_chat('{chat.cid}') later to retrieve it."
                    )

                if not (is_completed or is_final_chunk) or is_thinking or is_queueing:
                    logger.debug(
                        f"Stream interrupted (completed={is_completed}, final_chunk={is_final_chunk}, thinking={is_thinking}, queueing={is_queueing}). "
                        "Polling again..."
                    )
                    raise APIError("Stream interrupted or truncated.")

        except ReadTimeout:
            raise TimeoutError(
                "The request timed out while waiting for Gemini to respond. This often happens with very long prompts "
                "or complex file analysis. Try increasing the 'timeout' value when initializing GeminiClient."
            )
        except (GeminiError, APIError):
            raise
        except Exception as e:
            cause = e.__cause__ or e.__context__
            logger.warning(
                f"Stream error: {type(e).__name__}: {e!r}; "
                f"cause={type(cause).__name__}: {cause!r}; "
                f"chat.cid={getattr(chat, 'cid', None)!r}, "
                f"model={getattr(model, 'model_name', model)!r}"
                if cause else
                f"Stream error: {type(e).__name__}: {e!r}; "
                f"chat.cid={getattr(chat, 'cid', None)!r}, "
                f"model={getattr(model, 'model_name', model)!r}"
            )
            raise APIError(f"Failed to parse response body: {e}")
        finally:
            pass

    # Known tool triplets from qpEbW quota response
    _QUOTA_TOOL_NAMES = {
        (6, 6, 3): "deep_think",
        (6, 4, 3): "pro",
        (6, 15, 3): "flash_thinking",
    }

    async def check_quota(self) -> dict[str, dict]:
        """
        Query per-tool usage quotas from the server (best-effort).

        Returns a dict keyed by tool name with ``remaining``, ``limit``,
        and ``reset_timestamp`` (epoch seconds) for each tool.
        Returns empty dict if the server doesn't provide quota data
        (e.g., no prior StreamGenerate in this session).

        Known tools:
            - ``deep_think``: Deep Think 3.1 (10/day on Ultra)
            - ``pro``: Gemini 3.1 Pro (500/day on Ultra)
            - ``flash_thinking``: Flash Thinking (1,500/day on Ultra)

        Example return::

            {
                "deep_think": {"remaining": 8, "limit": 10, "reset_timestamp": 1776029977},
                "pro": {"remaining": 490, "limit": 500, "reset_timestamp": 1776029977},
                "flash_thinking": {"remaining": 1500, "limit": 1500, "reset_timestamp": 1776029977},
            }
        """
        payload = RPCData(
            rpcid=GRPC.CHECK_QUOTA,
            payload='[[[1,4],[6,6],[1,15]]]',
        )
        try:
            response = await self._batch_execute(
                [payload],
                headers={
                    "x-goog-ext-525001261-jspb": '[1,null,null,null,null,null,null,null,[4]]',
                    "x-goog-ext-73010989-jspb": "[0]",
                },
            )
        except Exception as e:
            logger.debug(f"check_quota request failed: {e}")
            return {}

        body = response.text
        if body.startswith(")]}'"):
            body = body[4:]

        result = {}
        try:
            parsed = json.loads(body.split("\n")[1])
            data_str = parsed[0][2]
            if data_str is None:
                return {}
            inner = json.loads(data_str)
            for entry in inner[0]:
                tool_triplet = tuple(entry[0])
                name = self._QUOTA_TOOL_NAMES.get(tool_triplet, f"unknown_{tool_triplet}")
                result[name] = {
                    "remaining": entry[5],
                    "limit": entry[4],
                    "reset_timestamp": entry[3][0] if isinstance(entry[3], list) else None,
                }
        except (json.JSONDecodeError, IndexError, TypeError, KeyError) as e:
            logger.debug(f"check_quota parse failed: {e}")
        return result

    def start_chat(self, **kwargs) -> "ChatSession":
        """
        Returns a `ChatSession` object attached to this client.

        Parameters
        ----------
        kwargs: `dict`, optional
            Additional arguments which will be passed to the chat session.
            Refer to `gemini_webapi.ChatSession` for more information.

        Returns
        -------
        :class:`ChatSession`
            Empty chat session object for retrieving conversation history.
        """

        return ChatSession(geminiclient=self, **kwargs)

    @running(retry=2)
    async def _batch_execute(self, payloads: list[RPCData], **kwargs) -> Response:
        """
        Execute a batch of requests to Gemini API.

        Parameters
        ----------
        payloads: `list[RPCData]`
            List of `gemini_webapi.types.RPCData` objects to be executed.
        kwargs: `dict`, optional
            Additional arguments which will be passed to the post request.
            Refer to `curl_cffi.requests.AsyncSession` for more information.

        Returns
        -------
        :class:`curl_cffi.requests.Response`
            Response object containing the result of the batch execution.
        """

        _reqid = self._reqid
        self._reqid += 100000

        try:
            params: dict[str, Any] = {
                "rpcids": ",".join([p.rpcid for p in payloads]),
                "_reqid": _reqid,
                "rt": "c",
                "source-path": "/app",
            }
            if self.build_label:
                params["bl"] = self.build_label
            if self.session_id:
                params["f.sid"] = self.session_id

            response = await self.client.post(
                Endpoint.BATCH_EXEC,
                params=params,
                data={
                    "at": self.access_token,
                    "f.req": json.dumps(
                        [[payload.serialize() for payload in payloads]]
                    ).decode("utf-8"),
                },
                **kwargs,
            )
        except ReadTimeout:
            raise TimeoutError(
                "The request timed out while waiting for Gemini to respond. This often happens with very long prompts "
                "or complex file analysis. Try increasing the 'timeout' value when initializing GeminiClient."
            )

        if response.status_code != 200:
            await self.close()
            raise APIError(
                f"Batch execution failed with status code {response.status_code}"
            )

        if self.client:
            self.cookies.update(self.client.cookies)

        return response


class ChatSession:
    """
    Chat data to retrieve conversation history. Only if all 3 ids are provided will the conversation history be retrieved.

    Parameters
    ----------
    geminiclient: `GeminiClient`
        Async client interface for gemini.google.com.
    metadata: `list[str]`, optional
        List of chat metadata `[cid, rid, rcid]`, can be shorter than 3 elements, like `[cid, rid]` or `[cid]` only.
    cid: `str`, optional
        Chat id, if provided together with metadata, will override the first value in it.
    rid: `str`, optional
        Reply id, if provided together with metadata, will override the second value in it.
    rcid: `str`, optional
        Reply candidate id, if provided together with metadata, will override the third value in it.
    model: `Model | str | dict`, optional
        Specify the model to use for generation.
        Pass either a `gemini_webapi.constants.Model` enum or a model name string to use predefined models.
        Pass a dictionary to use custom model header strings ("model_name" and "model_header" keys must be provided).
    gem: `Gem | str`, optional
        Specify a gem to use as system prompt for the chat session.
        Pass either a `gemini_webapi.types.Gem` object or a gem id string.
    """

    __slots__ = [
        "__metadata",
        "geminiclient",
        "last_output",
        "model",
        "gem",
    ]

    def __init__(
        self,
        geminiclient: GeminiClient,
        metadata: list[str | None] | None = None,
        cid: str | None = None,  # chat id
        rid: str | None = None,  # reply id
        rcid: str | None = None,  # reply candidate id
        model: Model | str | dict = Model.UNSPECIFIED,
        gem: Gem | str | None = None,
    ):
        self.__metadata: list[str | None] = [
            "",
            "",
            "",
            None,
            None,
            None,
            None,
            None,
            None,
            "",
        ]
        self.geminiclient: GeminiClient = geminiclient
        self.last_output: ModelOutput | None = None
        self.model: Model | str | dict = model
        self.gem: Gem | str | None = gem

        if metadata:
            self.metadata = metadata
        if cid:
            self.cid = cid
        if rid:
            self.rid = rid
        if rcid:
            self.rcid = rcid

    def __str__(self):
        return f"ChatSession(cid='{self.cid}', rid='{self.rid}', rcid='{self.rcid}')"

    __repr__ = __str__

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        # update conversation history when last output is updated
        if name == "last_output" and isinstance(value, ModelOutput):
            self.metadata = value.metadata
            self.rcid = value.rcid

    async def send_message(
        self,
        prompt: str,
        files: list[str | Path | bytes | io.BytesIO] | None = None,
        temporary: bool = False,
        **kwargs,
    ) -> ModelOutput:
        """
        Generates contents with prompt.
        Use as a shortcut for `GeminiClient.generate_content(prompt, files, self)`.

        Parameters
        ----------
        prompt: `str`
            Text prompt provided by user.
        files: `list[str | Path | bytes | io.BytesIO]`, optional
            List of file paths or byte streams to be attached.
        temporary: `bool`, optional
            If set to `True`, the ongoing conversation will not show up in Gemini history.
            Switching temporary mode within a chat session will clear the previous context
            and create a new chat session under the hood.
        kwargs: `dict`, optional
            Additional arguments which will be passed to the post request.
            Refer to `curl_cffi.requests.AsyncSession` for more information.

        Returns
        -------
        :class:`ModelOutput`
            Output data from gemini.google.com.

        Raises
        ------
        `AssertionError`
            If prompt is empty.
        `gemini_webapi.TimeoutError`
            If request timed out.
        `gemini_webapi.GeminiError`
            If no reply candidate found in response.
        `gemini_webapi.APIError`
            - If request failed with status code other than 200.
            - If response structure is invalid and failed to parse.
        """

        return await self.geminiclient.generate_content(
            prompt=prompt,
            files=files,
            model=self.model,
            gem=self.gem,
            chat=self,
            temporary=temporary,
            **kwargs,
        )

    async def send_message_stream(
        self,
        prompt: str,
        files: list[str | Path | bytes | io.BytesIO] | None = None,
        temporary: bool = False,
        **kwargs,
    ) -> AsyncGenerator[ModelOutput, None]:
        """
        Generates contents with prompt in streaming mode within this chat session.

        This is a shortcut for `GeminiClient.generate_content_stream(prompt, files, self)`.
        The session's metadata and conversation history are automatically managed.

        Parameters
        ----------
        prompt: `str`
            Text prompt provided by user.
        files: `list[str | Path | bytes | io.BytesIO]`, optional
            List of file paths or byte streams to be attached.
        temporary: `bool`, optional
            If set to `True`, the ongoing conversation will not show up in Gemini history.
            Switching temporary mode within a chat session will clear the previous context
            and create a new chat session under the hood.
        kwargs: `dict`, optional
            Additional arguments passed to the streaming request.

        Yields
        ------
        :class:`ModelOutput`
            Partial output data containing text deltas.
        """

        async for output in self.geminiclient.generate_content_stream(
            prompt=prompt,
            files=files,
            model=self.model,
            gem=self.gem,
            chat=self,
            temporary=temporary,
            **kwargs,
        ):
            yield output

    def choose_candidate(self, index: int) -> ModelOutput:
        """
        Choose a candidate from the last `ModelOutput` to control the ongoing conversation flow.

        Parameters
        ----------
        index: `int`
            Index of the candidate to choose, starting from 0.

        Returns
        -------
        :class:`ModelOutput`
            Output data of the chosen candidate.

        Raises
        ------
        `ValueError`
            If no previous output data found in this chat session, or if index exceeds the number of candidates in last model output.
        """

        if not self.last_output:
            raise ValueError("No previous output data found in this chat session.")

        if index >= len(self.last_output.candidates):
            raise ValueError(
                f"Index {index} exceeds the number of candidates in last model output."
            )

        self.last_output.chosen = index
        self.rcid = self.last_output.rcid
        return self.last_output

    @property
    def metadata(self):
        return self.__metadata

    @metadata.setter
    def metadata(self, value: list[str]):
        if not isinstance(value, list):
            return

        # Update only non-None elements to preserve existing CID/RID/RCID/Context
        for i, val in enumerate(value):
            if i < 10 and val is not None:
                self.__metadata[i] = val

    @property
    def cid(self):
        return self.__metadata[0]

    @cid.setter
    def cid(self, value: str):
        self.__metadata[0] = value

    @property
    def rcid(self):
        return self.__metadata[2]

    @rcid.setter
    def rcid(self, value: str):
        self.__metadata[2] = value

    @property
    def rid(self):
        return self.__metadata[1]

    @rid.setter
    def rid(self, value: str):
        self.__metadata[1] = value
