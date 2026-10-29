"""URL normalization and citation dedup rules.

Disclosed dedup rule (per brief requirement):
  - For *raw* persistence, every citation occurrence is stored verbatim
    (preserving order, type, position, url).
  - For *analysis*, "unique citations per response" = DISTINCT
    (response_id, normalized_url) where normalized_url strips the URL
    fragment (#...) and known tracking query parameters
    (utm_*, gclid, fbclid, ref, source, mc_cid, mc_eid).
  - Across responses, "citation recurrence" counts the same
    normalized_url appearing in >1 response_id.

URL fragments and tracking parameters are preserved in the raw `url`
column. Only `normalized_url` is computed.
"""
from __future__ import annotations

from urllib.parse import parse_qsl, urlsplit, urlunsplit

# Tracking params stripped during normalization. Keep this list small
# and obvious — every name removed must be documented here.
_TRACKING_PARAMS = frozenset(
    {
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "gclid",
        "fbclid",
        "msclkid",
        "ref",
        "source",
        "mc_cid",
        "mc_eid",
        "_hsenc",
        "_hsmi",
        "igshid",
    }
)


def normalize_url(url: str) -> str:
    """Return a stable URL key for deduplication.

    Steps:
      1. Lowercase the scheme + host (paths are case-sensitive).
      2. Strip URL fragment.
      3. Drop tracking query parameters (defined above).
      4. Sort remaining query parameters for canonical order.
      5. Strip trailing slash on path (except for root).
      6. Drop default ports (:80, :443).

    Empty/None input returns the empty string.
    """
    if not url or not url.strip():
        return ""
    url = url.strip()
    try:
        parts = urlsplit(url)
    except ValueError:
        return url

    scheme = parts.scheme.lower()
    netloc = parts.netloc.lower()
    path = parts.path or ""

    # Strip default ports
    if scheme == "http" and netloc.endswith(":80"):
        netloc = netloc[:-3]
    elif scheme == "https" and netloc.endswith(":443"):
        netloc = netloc[:-4]

    # Strip trailing slash except for root
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")

    # Drop tracking params, sort the rest
    kept = sorted(
        (k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if k.lower() not in _TRACKING_PARAMS
    )
    query = "&".join(f"{k}={v}" for k, v in kept)

    # Fragment intentionally dropped
    return urlunsplit((scheme, netloc, path, query, ""))


def domain_of(url: str) -> str | None:
    """Return the registered domain (host without leading www.) or None."""
    if not url:
        return None
    try:
        host = urlsplit(url).netloc.lower()
    except ValueError:
        return None
    if not host:
        return None
    if host.startswith("www."):
        host = host[4:]
    # Strip port
    return host.split(":", 1)[0] or None


# Known Kasparro domains — used to flag has_kasparro_citation.
KASPARRO_DOMAINS = frozenset({"kasparro.com", "www.kasparro.com"})


def is_kasparro_domain(host: str | None) -> bool:
    return bool(host) and (host in KASPARRO_DOMAINS or host.endswith(".kasparro.com"))
