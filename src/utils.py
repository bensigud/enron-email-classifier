"""
Shared utility functions.
"""


def truncate_email(address: str) -> str:
    """Extract the name part from an email address."""
    return address.split("@")[0].replace(".", " ").title() if address else ""
