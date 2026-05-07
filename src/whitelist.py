"""
whitelist.py
------------
Filters out trusted CDN and known-good domains to reduce false positives.
"""

import tldextract
import pandas as pd

# Common CDN and trusted domains
CDN_WHITELIST = {
    "cloudflare.com", "akamaiedge.net", "fastly.net", "amazonaws.com",
    "azureedge.net", "cloudfront.net", "googleusercontent.com",
    "googlevideo.com", "gstatic.com", "googleapis.com",
    "akamai.net", "edgesuite.net", "llnwd.net", "footprint.net",
    "microsoft.com", "office.com", "windows.com", "live.com",
    "apple.com", "icloud.com", "akadns.net", "edgekey.net",
}


def is_whitelisted(domain: str) -> bool:
    """Check if a domain belongs to a trusted CDN or provider."""
    extracted = tldextract.extract(domain)
    registered_domain = f"{extracted.domain}.{extracted.suffix}"
    return registered_domain in CDN_WHITELIST


def filter_whitelist(df: pd.DataFrame, domain_col: str = "query_name") -> pd.DataFrame:
    """Remove whitelisted domains from the dataframe."""
    mask = df[domain_col].apply(lambda d: not is_whitelisted(str(d)))
    filtered = df[mask].reset_index(drop=True)
    removed = len(df) - len(filtered)
    print(f"Whitelist filter: removed {removed} trusted CDN records ({len(filtered)} remaining)")
    return filtered


if __name__ == "__main__":
    test_domains = [
        "example.cloudfront.net",
        "suspicious.xyz123abc.com",
        "data.googleapis.com",
        "exfil.attacker.io",
    ]
    for d in test_domains:
        print(f"{d:40s} -> {'WHITELISTED' if is_whitelisted(d) else 'INSPECT'}")
