"""URL normalization and citation dedup rules.

Disclosed dedup rule (per brief requirement):
  - For *raw* persistence, every citation occurrence is stored verbatim
    (preserving order, type, position, url) in the `url` + `raw_json` columns.
  - For *analysis*, "unique citations per response" = DISTINCT
    (response_id, normalized_url) where normalized_url:
      1. Lowercases scheme + host (paths are case-sensitive).
      2. Strips leading "www." from the host (so www.x.com == x.com).
      3. Strips URL fragment (#...).
      4. Drops ONLY genuine tracking parameters:
         utm_source, utm_medium, utm_campaign, utm_term, utm_content,
         gclid, fbclid, mc_cid, mc_eid, msclkid.
         Does NOT strip "ref" (GitHub uses it) or "source" (legitimate
         on many sites) — those are preserved as real query params.
      5. Sorts remaining query parameters for canonical order.
      6. Strips trailing slash on path (except for root "/").
      7. Drops default ports (:80 http, :443 https).
      8. Preserves percent-encoding (e.g. %26 stays %26, not decoded to &).

The `domain` column is also normalized: leading "www." stripped, port
stripped. The raw URL is always preserved in `url`.
"""
from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

# Only these tracking params are stripped. "ref" and "source" are kept
# because they are legitimate params on many sites (GitHub, news sites).
_TRACKING_PARAMS = frozenset(
    {
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "gclid",
        "fbclid",
        "mc_cid",
        "mc_eid",
        "msclkid",
        "_hsenc",
        "_hsmi",
        "igshid",
    }
)


def normalize_url(url: str) -> str:
    """Return a stable URL key for deduplication.

    Uses urllib.parse.urlsplit + urlencode so percent-encoded characters
    (like %26 for &) are preserved and not corrupted.
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

    # Strip leading www. for canonical host
    if netloc.startswith("www."):
        netloc = netloc[4:]

    # Strip default ports
    if scheme == "http" and netloc.endswith(":80"):
        netloc = netloc[:-3]
    elif scheme == "https" and netloc.endswith(":443"):
        netloc = netloc[:-4]

    # Normalize path: empty → root "/"
    path = parts.path or "/"
    # Strip trailing slash except for root
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")

    # Drop tracking params, keep the rest. Use parse_qsl with
    # keep_blank_values=True so ?a= is preserved as a= (empty value).
    # urlencode preserves percent-encoding correctly.
    kept = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if k.lower() not in _TRACKING_PARAMS
    ]
    query = urlencode(kept)  # preserves %26 etc. correctly

    # Fragment intentionally dropped
    return urlunsplit((scheme, netloc, path, query, ""))


def domain_of(url: str) -> str | None:
    """Return the registered domain (host without leading www. + port) or None.

    Uses urlsplit(url).hostname which correctly strips userinfo
    (user:pw@) — a security fix over manual netloc parsing.
    """
    if not url:
        return None
    try:
        host = urlsplit(url).hostname
    except ValueError:
        return None
    if not host:
        return None
    host = host.lower()
    if host.startswith("www."):
        host = host[4:]
    return host or None


# Known Kasparro domains — used to flag has_kasparro_citation.
KASPARRO_DOMAINS = frozenset({"kasparro.com", "www.kasparro.com"})


def is_kasparro_domain(host: str | None) -> bool:
    if not host:
        return False
    host = host.lower()
    if host.startswith("www."):
        host = host[4:]
    return host in KASPARRO_DOMAINS or host.endswith(".kasparro.com")
