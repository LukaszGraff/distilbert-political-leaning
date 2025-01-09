# import re

# class SelfReportDetector:
#     def __init__(self):
#         # "As a" prefix requires ideology followed by either 'I' or 'we', 
#         # allowing an optional comma and spaces before 'I' or 'we'.
#         self.as_a_prefix = r"As\ a"

#         # Other prefix patterns (no strict 'I' or 'we' after)
#         self.other_prefixes = [
#             r"I\s+(?:consider|view|see|identify)\s+myself(?:\s+as)?",
#             r"I\s+am"
#         ]

#         # Left-leaning ideologies with plural forms
#         self.left_ideologies = [
#             r"left[-\s]?wing(?:ers?)?",
#             r"socialist(?:s)?",
#             r"communist(?:s)?",
#             r"anarchist(?:s)?",
#             r"left[-\s]?libertarian(?:s)?"
#         ]

#         # Right-leaning ideologies with plural forms
#         self.right_ideologies = [
#             r"right[-\s]?wing(?:ers?)?",
#             r"capitalist(?:s)?"
#             r"conservatist(?:s)?"
#             r"right[-\s]?libertarian(?:s)?"
#             r"fundamentalist(?:s)?"
#         ]

#         all_ideologies = self.left_ideologies + self.right_ideologies

#         # Construct the pattern for "As a" prefix with optional comma before I or we:
#         # As a (ideology) [optional comma] (I|we)
#         as_a_pattern = rf"""
#         {self.as_a_prefix}          # "As a"
#         \s+
#         (?P<ideology>
#             {"|".join(all_ideologies)}
#         )                            # ideologies
#         \s*(?:,\s*)?                 # optional comma and spaces
#         (I|we)\b                     # must be followed by I or we
#         """

#         # Construct the pattern for other prefixes (no strict "I" or "we" after):
#         other_prefixes_pattern = rf"""
#         (?:
#             {"|".join(self.other_prefixes)}
#         )
#         \s+
#         (?P<ideology>
#             {"|".join(all_ideologies)}
#         )
#         \b
#         """

#         # Compile the regexes with IGNORECASE and VERBOSE
#         self.compiled_as_a_pattern = re.compile(as_a_pattern, re.IGNORECASE | re.VERBOSE)
#         self.compiled_other_pattern = re.compile(other_prefixes_pattern, re.IGNORECASE | re.VERBOSE)

#         # Keyword lists for heuristic detection
#         self.left_keywords = ["universal healthcare", "social justice", "climate change", "progressive", "equality"]
#         self.right_keywords = ["free market", "tax cuts", "gun rights", "nationalism", "authoritarian"]
#         self.center_keywords = ["bipartisan", "moderate", "third way"]

#     def preprocess(self, text: str) -> str:
#         """Normalize and clean the text."""
#         text = re.sub(r"http[s]?://\S+", "", text)  # Remove URLs
#         text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
#         text = text.lower().strip()                   # Convert to lowercase
#         return text

#     def detect(self, texts: list) -> list:
#         """
#         Detect if a given list of texts includes self-declared left or right ideologies.
#         Returns a list of 'left', 'right', or 'none' for each text.

#         Conditions:
#           - "As a" prefix requires the sequence "As a <ideology> [optional comma] I/we"
#           - Other prefixes don't have this requirement.
#         """
#         results = []
#         for text in texts:
#             # Check the "As a" pattern first
#             preprocessed_text = self.preprocess(text)
#             match = self.compiled_as_a_pattern.search(preprocessed_text)
#             if match:
#                 ideology = match.group('ideology')
#                 results.append(self._determine_side(ideology))
#                 continue

#             # Check the other prefix pattern
#             match = self.compiled_other_pattern.search(preprocessed_text)
#             if match:
#                 ideology = match.group('ideology')
#                 results.append(self._determine_side(ideology))
#                 continue

#             results.append(self._keyword_based_detection(preprocessed_text))
#         return results

#     def _determine_side(self, ideology: str) -> str:
#         """
#         Determine if the matched ideology is left or right leaning.
#         """
#         # Normalize for checking
#         ideology = ideology.lower()
#         if any(ideo in ideology for ideo in ['left-wing', 'left wing', 'leftwing', 'socialist']):
#             return 'left'
#         elif any(ideo in ideology for ideo in ['right-wing', 'right wing', 'rightwing', 'capitalist']):
#             return 'right'
#         return 'none'


#     def _keyword_based_detection(self, text: str) -> str:
#         """Heuristic detection based on keywords."""
#         if any(keyword in text for keyword in self.left_keywords):
#             return 'left'
#         elif any(keyword in text for keyword in self.right_keywords):
#             return 'right'
#         elif any(keyword in text for keyword in self.center_keywords):
#             return 'center'
#         return 'none'

# # # Example usage
# # if __name__ == "__main__":
# #     detector = SelfReportDetector()
# #     text_samples = [
# #         "As a socialist, we believe in universal healthcare.",
# #         "As a socialists I know the struggle.",
# #         "As a left-winger I always vote green.",
# #         "I consider myself as left-wing.",
# #         "I view myself right wing and strongly support market freedoms.",
# #         "I am capitalist at heart.",
# #         "As a right-wing individuals, we support strict immigration laws.",
# #         "I consider myself socialist.",
# #         "I identify myself as a left wing economist."
# #     ]

# #     for text in text_samples:
# #         ideology = detector.detect_ideology(text)
# #         print(f"'{text}' -> Detected ideology: {ideology}")

# TEST VERSION

#!/usr/bin/env python3

"""
detect_self_reporting.py

This script flags Reddit posts that contain explicit first-person self-reports
of political orientation (e.g., "I am a leftist", "I used to be a conservative", etc.).
It uses a comprehensive set of regex patterns combined with a list of known political terms.
"""

import re

###############################################################################
# 1. Define lists of political-leaning synonyms
###############################################################################

LEFT_TERMS = [
    "leftist", "liberal", "progressive", "democrat", "socialist"
]
RIGHT_TERMS = [
    "conservative", "republican", "gop", "maga", "trumpist"
]
CENTER_TERMS = [
    "centrist", "moderate", "independent", "libertarian"
]

ALL_TERMS = LEFT_TERMS + RIGHT_TERMS + CENTER_TERMS

# Join them into one big alternation group for regex: (leftist|liberal|conservative|...)
terms_pattern = "|".join(ALL_TERMS)

###############################################################################
# 2. Build regex patterns for first-person self-identification
###############################################################################
# Explanation of each block:
# - We focus on the author referencing themselves (I, my, etc.)
# - Then referencing a political orientation from our known set of terms.
# - Each pattern uses word boundaries (\b) and optional articles (a|an).
# - We use re.IGNORECASE later to match any case style.

regex_patterns = [
    # (1) I am a/an X
    rf"\bi\s+am\s+(?:a|an)?\s*(?:{terms_pattern})\b",

    # (2) I'm a/an X
    rf"\bi'?m\s+(?:a|an)?\s*(?:{terms_pattern})\b",

    # (3) I consider myself X
    rf"\bi\s+consider\s+myself\s+(?:{terms_pattern})\b",

    # (4) I identify as X
    rf"\bi\s+identify\s+as\s+(?:a|an)?\s*(?:{terms_pattern})\b",

    # (5) As a X, ...
    rf"\bas\s+(?:a|an)?\s*(?:{terms_pattern})\b",

    # (6) My stance/position/leaning/ideology is X
    rf"\bmy\s+(?:stance|position|leaning|ideology)\s+is\s+(?:{terms_pattern})\b",

    # (7) I used to be/ I was X
    rf"\bi\s+(?:used\s+to\s+be|was)\s+(?:a|an)?\s*(?:{terms_pattern})\b",

    # (8) I side with / I align with X
    rf"\bi\s+(?:side|align)\s+with\s+(?:the\s+)?(?:{terms_pattern})\b",

    # (9) I'd label myself as / I label myself as X
    rf"\bi(?:'?d)?\s+label\s+myself\s+as\s+(?:a|an)?\s*(?:{terms_pattern})\b",

    # (10) I describe myself as X
    rf"\bi\s+describe\s+myself\s+as\s+(?:a|an)?\s*(?:{terms_pattern})\b",

    # (11) I call myself X
    rf"\bi\s+call\s+myself\s+(?:a|an)?\s*(?:{terms_pattern})\b",

    # (12) My personal politics/beliefs are X
    rf"\bmy\s+(?:personal\s+)?(?:politics|beliefs)\s+are\s+(?:{terms_pattern})\b",

    # (13) My political orientation/identity is X
    rf"\bmy\s+political\s+(?:orientation|identity)\s+is\s+(?:{terms_pattern})\b",

    # (14) I'm on the left/right/center
    rf"\bi'?m\s+on\s+the\s+(?:{terms_pattern})\b",

    # (15) I'm left-leaning / right-leaning / center-leaning
    rf"\bi'?m\s+(?:left-leaning|right-leaning|center-leaning)\b",

    # (16) My ideology is X
    rf"\bmy\s+ideology\s+is\s+(?:{terms_pattern})\b",
]

# Combine into one big pattern with OR, and compile for performance (ignore case)
combined_pattern = re.compile("|".join(regex_patterns), re.IGNORECASE)

###############################################################################
# 3. Classification function
###############################################################################
def is_self_reporting(post_text: str) -> bool:
    """
    Returns True if the post contains explicit first-person self-report of
    political leaning, otherwise False.
    """
    return bool(combined_pattern.search(post_text))

###############################################################################
# 4. Example usage
###############################################################################
# if __name__ == "__main__":
#     sample_posts = [
#         "As a leftist, I'm all for universal healthcare.",
#         "I think we should have universal healthcare. No direct statement about me here.",
#         "I’m a conservative, but I actually agree on some social policies.",
#         "My stance is liberal, so I support progressive taxes.",
#         "He said I'm just a leftist, but I never said that. (Should NOT be flagged)",
#         "I identify as a moderate, and I think extremes are harmful.",
#         "People keep calling me liberal, but that's them saying it, not me.",
#         "I used to be a conservative, but I've moved more to the left.",
#         "I side with Democrats on social issues.",
#         "I consider myself a libertarian with progressive leanings.",
#         "I'm on the right, but not far right.",
#         "I'm left-leaning on social issues.",
#         "I describe myself as a socialist with some moderate views.",
#         "My personal politics are moderate, generally.",
#     ]

#     for post in sample_posts:
#         flagged = is_self_reporting(post)
#         status = "FLAGGED" if flagged else "OK"
#         print(f"{status}: {post}")
