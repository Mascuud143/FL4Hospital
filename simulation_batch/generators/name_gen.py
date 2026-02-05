import random

# Name parts
prefixes = ["Ka", "El", "Ro", "Mi", "Sa", "Jo", "An", "Lu", "Ti", "Be", "Vi"]
cores = ["lin", "var", "dor", "mar", "nel", "tis", "ven", "ric", "sol", "zen", "cal"]
endings = ["a", "en", "ix", "or", "us", "ia", "an", "el", "is", "on", "ar"]

def generate_name():
    return (
        random.choice(prefixes)
        + random.choice(cores)
        + random.choice(endings)
    )


