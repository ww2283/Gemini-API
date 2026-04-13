import os
import time
from pathlib import Path

from curl_cffi.requests import AsyncSession, Cookies

from ..constants import Endpoint, Headers
from ..exceptions import AuthError


def _extract_cookie_value(cookies: Cookies, name: str) -> str | None:
    """Extract a cookie value from a curl_cffi Cookies jar."""
    for cookie in cookies.jar:
        if cookie.name == name:
            return cookie.value
    return None


async def rotate_1psidts(
    client: AsyncSession, verbose: bool = False,
) -> tuple[str | None, Cookies | None]:
    """
    Refresh the __Secure-1PSIDTS cookie and store the refreshed cookie value in cache file.

    Parameters
    ----------
    client : `curl_cffi.requests.AsyncSession`
        The shared async session (with cookies already set).
    verbose: `bool`, optional
        If `True`, will print more infomation in logs.

    Returns
    -------
    `tuple[str | None, Cookies | None]`
        New value of the __Secure-1PSIDTS cookie and the full updated cookies jar.

    Raises
    ------
    `gemini_webapi.AuthError`
        If request failed with 401 Unauthorized.
    `curl_cffi.requests.exceptions.HTTPError`
        If request failed with other status codes.
    """

    path = (
        (GEMINI_COOKIE_PATH := os.getenv("GEMINI_COOKIE_PATH"))
        and Path(GEMINI_COOKIE_PATH)
        or (Path(__file__).parent / "temp")
    )
    path.mkdir(parents=True, exist_ok=True)

    secure_1psid = _extract_cookie_value(client.cookies, "__Secure-1PSID")
    if not secure_1psid:
        return None, None

    filename = f".cached_1psidts_{secure_1psid}.txt"
    path = path / filename

    # Check if the cache file was modified in the last minute to avoid 429 Too Many Requests
    if path.is_file() and time.time() - os.path.getmtime(path) <= 60:
        return path.read_text(), None

    response = await client.post(
        url=Endpoint.ROTATE_COOKIES,
        headers=Headers.ROTATE_COOKIES.value,
        content='[000,"-0000000000000000000"]',
    )
    if response.status_code == 401:
        raise AuthError
    response.raise_for_status()

    new_1psidts = _extract_cookie_value(response.cookies, "__Secure-1PSIDTS")
    if new_1psidts:
        path.write_text(new_1psidts)
        path.chmod(0o600)
        return new_1psidts, response.cookies

    return None, response.cookies
