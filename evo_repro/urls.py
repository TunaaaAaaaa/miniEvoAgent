from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


REDACTED_VALUE = "***"
_SENSITIVE_QUERY_KEYS = {
    "access_token",
    "api_key",
    "apikey",
    "client_secret",
    "key",
    "pass",
    "passwd",
    "password",
    "pwd",
    "secret",
    "sslpassword",
    "token",
}
_SENSITIVE_QUERY_SUFFIXES = ("_api_key", "_password", "_secret", "_token")


def redact_url_secrets(url: str | None) -> str | None:
    """Redact user-info passwords and secret-looking URL query parameters."""

    if not url:
        return url

    parts = urlsplit(url)
    netloc = _redact_user_info(parts.netloc)
    query = urlencode(
        [
            (key, REDACTED_VALUE if _is_sensitive_query_key(key) else value)
            for key, value in parse_qsl(parts.query, keep_blank_values=True)
        ],
        doseq=True,
        safe="*",
    )
    return urlunsplit((parts.scheme, netloc, parts.path, query, parts.fragment))


def _redact_user_info(netloc: str) -> str:
    if "@" not in netloc:
        return netloc
    user_info, host_info = netloc.rsplit("@", 1)
    if ":" not in user_info:
        return netloc
    username, _ = user_info.split(":", 1)
    return f"{username}:{REDACTED_VALUE}@{host_info}"


def _is_sensitive_query_key(key: str) -> bool:
    normalized = key.casefold().replace("-", "_")
    return normalized in _SENSITIVE_QUERY_KEYS or normalized.endswith(
        _SENSITIVE_QUERY_SUFFIXES
    )
