'''
The activity a cross-reference redirects (Req 8): its text before the redirection.

"Growing soybeans--are classified in Industry 111110" sends soybean growing to 111110, and its
activity phrase is "Growing soybeans". The redirection table's build (``data.redirections``) takes
each row's phrase from here, and the bundle loader (``supervision.artifacts``) recomputes every
non-withheld row's phrase from here and refuses a mismatch. The module imports only the standard
library, so the loader can import it without the build.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import re
from typing import Optional

# "Growing soybeans--are classified in ...", "Establishments ... are classified in ...", and the
# source's one misspelling, "are lclassified"
_REDIRECTION = re.compile(r'(?:--|\s+)(?=(?:are|is)\s+l?(?:classified|included)\b)')

# -------------------------------------------------------------------------------------------------
# The phrase
# -------------------------------------------------------------------------------------------------

def activity_phrase(text: str) -> Optional[str]:
    '''The activity a cross-reference redirects: its text before the redirection, or None.'''

    match = _REDIRECTION.search(text)
    if match is None:
        return None
    return text[:match.start()].strip() or None
