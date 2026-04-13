import os
import re
from pathlib import Path

from curl_cffi.requests import AsyncSession, Cookies, Response

from ..constants import Endpoint, Headers
from ..exceptions import AuthError
from .load_browser_cookies import load_browser_cookies
from .logger import logger


def _extract_cookie_value(cookies: Cookies, name: str) -> str | None:
    """Extract a cookie value from a curl_cffi Cookies jar."""
    for cookie in cookies.jar:
        if cookie.name == name:
            return cookie.value
    return None


async def _try_cookies(
    client: AsyncSession, cookies: dict | Cookies, verbose: bool = False,
) -> tuple[str | None, str | None, str | None, Cookies] | None:
    """Try a single cookie set. Returns (token, build_label, session_id, cookies) or None."""
    client.cookies.clear()
    if isinstance(cookies, Cookies):
        client.cookies.update(cookies)
    else:
        for k, v in cookies.items():
            client.cookies.set(k, v, domain=".google.com")

    try:
        response = await client.get(Endpoint.INIT, headers=Headers.GEMINI.value)
        response.raise_for_status()
    except Exception as e:
        if verbose:
            logger.debug(f"Init request failed: {e}")
        return None

    snlm0e = re.search(r'"SNlM0e":\s*"(.*?)"', response.text)
    cfb2h = re.search(r'"cfb2h":\s*"(.*?)"', response.text)
    fdrfje = re.search(r'"FdrFJe":\s*"(.*?)"', response.text)
    if snlm0e or cfb2h or fdrfje:
        return (
            snlm0e.group(1) if snlm0e else None,
            cfb2h.group(1) if cfb2h else None,
            fdrfje.group(1) if fdrfje else None,
            client.cookies,
        )
    return None


async def get_access_token(
    base_cookies: dict | Cookies,
    proxy: str | None = None,
    verbose: bool = False,
    verify: bool = True,
) -> tuple[str | None, str | None, str | None, Cookies, AsyncSession]:
    """
    Try each group of available cookies sequentially and return
    the value of "SNlM0e" as access token on the first successful request.

    Returns the live AsyncSession so the caller can reuse the same
    TLS connection (with Chrome impersonation) for subsequent requests.

    Parameters
    ----------
    base_cookies : `dict | curl_cffi.requests.Cookies`
        Base cookies to be used in the request.
    proxy: `str`, optional
        Proxy URL.
    verbose: `bool`, optional
        If `True`, will print more infomation in logs.
    verify: `bool`, optional
        Whether to verify SSL certificates.

    Returns
    -------
    `tuple[str | None, str | None, str | None, Cookies, AsyncSession]`
        By order: access token; build label; session id; cookies; live session.

    Raises
    ------
    `gemini_webapi.AuthError`
        If all requests failed.
    """

    client = AsyncSession(
        impersonate="chrome", proxy=proxy, allow_redirects=True, verify=verify
    )

    try:
        response = await client.get(Endpoint.GOOGLE)
    except Exception:
        response = None

    extra_cookies = Cookies()
    if response and response.status_code == 200:
        extra_cookies = response.cookies

    # Build list of cookie sets to try (sequentially — they share one session)
    cookie_sets: list[tuple[str, dict | Cookies]] = []

    if isinstance(base_cookies, Cookies):
        secure_1psid = _extract_cookie_value(base_cookies, "__Secure-1PSID")
        secure_1psidts = _extract_cookie_value(base_cookies, "__Secure-1PSIDTS")
    else:
        secure_1psid = base_cookies.get("__Secure-1PSID")
        secure_1psidts = base_cookies.get("__Secure-1PSIDTS")

    # 1. Base cookies
    if secure_1psid and secure_1psidts:
        jar = Cookies(extra_cookies)
        jar.update(base_cookies)
        cookie_sets.append(("base cookies", jar))
    elif verbose:
        logger.debug(
            "Skipping loading base cookies. Either __Secure-1PSID or __Secure-1PSIDTS is not provided."
        )

    # 2. Cached cookies
    cache_dir = (
        (GEMINI_COOKIE_PATH := os.getenv("GEMINI_COOKIE_PATH"))
        and Path(GEMINI_COOKIE_PATH)
        or (Path(__file__).parent / "temp")
    )

    if secure_1psid:
        filename = f".cached_1psidts_{secure_1psid}.txt"
        cache_file = cache_dir / filename
        if cache_file.is_file():
            cached_1psidts = cache_file.read_text()
            if cached_1psidts:
                jar = Cookies(extra_cookies)
                jar.update(base_cookies)
                jar.set("__Secure-1PSIDTS", cached_1psidts, domain=".google.com")
                cookie_sets.append(("cached cookies", jar))
            elif verbose:
                logger.debug("Skipping loading cached cookies. Cache file is empty.")
        elif verbose:
            logger.debug("Skipping loading cached cookies. Cache file not found.")
    else:
        valid_caches = 0
        cache_files = cache_dir.glob(".cached_1psidts_*.txt")
        for cache_file in cache_files:
            cached_1psidts = cache_file.read_text()
            if cached_1psidts:
                jar = Cookies(extra_cookies)
                psid = cache_file.stem[16:]
                jar.set("__Secure-1PSID", psid, domain=".google.com")
                jar.set("__Secure-1PSIDTS", cached_1psidts, domain=".google.com")
                cookie_sets.append((f"cached cookies ({psid[:8]}...)", jar))
                valid_caches += 1

        if valid_caches == 0 and verbose:
            logger.debug(
                "Skipping loading cached cookies. Cookies will be cached after successful initialization."
            )

    # 3. Browser cookies (if browser-cookie3 is installed)
    try:
        browser_cookies = load_browser_cookies(
            domain_name="google.com", verbose=verbose
        )
        if browser_cookies:
            for browser, cookies in browser_cookies.items():
                if browser_psid := cookies.get("__Secure-1PSID"):
                    if secure_1psid and secure_1psid != browser_psid:
                        if verbose:
                            logger.debug(
                                f"Skipping loading local browser cookies from {browser}. "
                                f"__Secure-1PSID does not match the one provided."
                            )
                        continue

                    local_cookies = {"__Secure-1PSID": browser_psid}
                    if browser_psidts := cookies.get("__Secure-1PSIDTS"):
                        local_cookies["__Secure-1PSIDTS"] = browser_psidts
                    if nid := cookies.get("NID"):
                        local_cookies["NID"] = nid
                    cookie_sets.append((f"browser ({browser})", local_cookies))
                    if verbose:
                        logger.debug(f"Loaded local browser cookies from {browser}")
    except ImportError:
        if verbose:
            logger.debug(
                "Skipping loading local browser cookies. Optional dependency 'browser-cookie3' is not installed."
            )
    except Exception as e:
        if verbose:
            logger.warning(f"Skipping loading local browser cookies. {e}")

    if not cookie_sets:
        raise AuthError(
            "No valid cookies available for initialization. Please pass __Secure-1PSID and __Secure-1PSIDTS manually."
        )

    # Try each cookie set sequentially (shared session — cannot run concurrently)
    for i, (source, cookies) in enumerate(cookie_sets):
        result = await _try_cookies(client, cookies, verbose=verbose)
        if result:
            if verbose:
                logger.debug(
                    f"Init attempt ({i + 1}/{len(cookie_sets)}) succeeded via {source}. Initializing client..."
                )
            access_token, build_label, session_id, valid_cookies = result
            return access_token, build_label, session_id, valid_cookies, client
        elif verbose:
            logger.debug(
                f"Init attempt ({i + 1}/{len(cookie_sets)}) via {source} failed. Cookies invalid."
            )

    await client.close()
    raise AuthError(
        "Failed to initialize client. SECURE_1PSIDTS could get expired frequently, please make sure cookie values are up to date. "
        f"(Failed initialization attempts: {len(cookie_sets)})"
    )
