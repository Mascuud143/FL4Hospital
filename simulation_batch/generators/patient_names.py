"""Generate synthetic patient names from predefined syllable lists."""

from __future__ import annotations

# Build patient names.
# - Picks a name prefix
# - Picks a middle syllable
# - Picks a name ending
# - Builds one name - generate_name()

import random


PREFIXES = ["Ka", "El", "Ro", "Mi", "Sa", "Jo", "An", "Lu", "Ti", "Be", "Vi"]
CORES = ["lin", "var", "dor", "mar", "nel", "tis", "ven", "ric", "sol", "zen", "cal"]
ENDINGS = ["a", "en", "ix", "or", "us", "ia", "an", "el", "is", "on", "ar"]


def generate_name() -> str:
    return random.choice(PREFIXES) + random.choice(CORES) + random.choice(ENDINGS)


__all__ = ["generate_name", "PREFIXES", "CORES", "ENDINGS"]
