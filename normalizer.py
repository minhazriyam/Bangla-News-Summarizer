# normalizer.py
import re

def normalize(s: str) -> str:
    if s is None:
        return ""
    s = s.replace("\u200c", "").replace("\u200d", "")  # zero-width
    s = re.sub(r"\s+", " ", s).strip()
    return s
BOILERPLATE_PATTERNS = [
    r"আরও( ?)?পড়(া|ুন).*?$",
    r"বিস্তারিত( ?)?[›»:].*$",
    r"শেয়ার( করুন)?[:।].*$",
    r"(ছবি|ফটো)[ :।-].*$",
    r"(সূত্র|Source)[:।-].*$",
    r"(Follow|ফলো) (us|করুন).*$",
    r"#\\w+",  # hashtags
]
BOILERPLATE_RE = re.compile("|".join(BOILERPLATE_PATTERNS), flags=re.IGNORECASE | re.MULTILINE)

def strip_boilerplate(s: str) -> str:
    if not s:
        return s
    s = BOILERPLATE_RE.sub("", s)
    # collapse whitespace again
    s = re.sub(r"\s+", " ", s).strip()
    return s