import re

class SelfReportDetector:
    def __init__(self):
        # "As a" prefix requires ideology followed by either 'I' or 'we', 
        # allowing an optional comma and spaces before 'I' or 'we'.
        self.as_a_prefix = r"As\ a"

        # Other prefix patterns (no strict 'I' or 'we' after)
        self.other_prefixes = [
            r"I\s+(?:consider|view|see|identify)\s+myself(?:\s+as)?",
            r"I\s+am"
        ]

        # Left-leaning ideologies with plural forms
        self.left_ideologies = [
            r"left[-\s]?wing(?:ers?)?",
            r"socialist(?:s)?"
        ]

        # Right-leaning ideologies with plural forms
        self.right_ideologies = [
            r"right[-\s]?wing(?:ers?)?",
            r"capitalist(?:s)?"
        ]

        all_ideologies = self.left_ideologies + self.right_ideologies

        # Construct the pattern for "As a" prefix with optional comma before I or we:
        # As a (ideology) [optional comma] (I|we)
        as_a_pattern = rf"""
        {self.as_a_prefix}          # "As a"
        \s+
        (?P<ideology>
            {"|".join(all_ideologies)}
        )                            # ideologies
        \s*(?:,\s*)?                 # optional comma and spaces
        (I|we)\b                     # must be followed by I or we
        """

        # Construct the pattern for other prefixes (no strict "I" or "we" after):
        other_prefixes_pattern = rf"""
        (?:
            {"|".join(self.other_prefixes)}
        )
        \s+
        (?P<ideology>
            {"|".join(all_ideologies)}
        )
        \b
        """

        # Compile the regexes with IGNORECASE and VERBOSE
        self.compiled_as_a_pattern = re.compile(as_a_pattern, re.IGNORECASE | re.VERBOSE)
        self.compiled_other_pattern = re.compile(other_prefixes_pattern, re.IGNORECASE | re.VERBOSE)

    def detect(self, texts: list) -> list:
        """
        Detect if a given list of texts includes self-declared left or right ideologies.
        Returns a list of 'left', 'right', or 'none' for each text.

        Conditions:
          - "As a" prefix requires the sequence "As a <ideology> [optional comma] I/we"
          - Other prefixes don't have this requirement.
        """
        results = []
        for text in texts:
            # Check the "As a" pattern first
            match = self.compiled_as_a_pattern.search(text)
            if match:
                ideology = match.group('ideology')
                results.append(self._determine_side(ideology))
                continue

            # Check the other prefix pattern
            match = self.compiled_other_pattern.search(text)
            if match:
                ideology = match.group('ideology')
                results.append(self._determine_side(ideology))
                continue

            results.append('none')
        return results

    def _determine_side(self, ideology: str) -> str:
        """
        Determine if the matched ideology is left or right leaning.
        """
        # Normalize for checking
        ideology = ideology.lower()
        if any(ideo in ideology for ideo in ['left-wing', 'left wing', 'leftwing', 'socialist']):
            return 'left'
        elif any(ideo in ideology for ideo in ['right-wing', 'right wing', 'rightwing', 'capitalist']):
            return 'right'
        return 'none'


# # Example usage
# if __name__ == "__main__":
#     detector = SelfReportDetector()
#     text_samples = [
#         "As a socialist, we believe in universal healthcare.",
#         "As a socialists I know the struggle.",
#         "As a left-winger I always vote green.",
#         "I consider myself as left-wing.",
#         "I view myself right wing and strongly support market freedoms.",
#         "I am capitalist at heart.",
#         "As a right-wing individuals, we support strict immigration laws.",
#         "I consider myself socialist.",
#         "I identify myself as a left wing economist."
#     ]

#     for text in text_samples:
#         ideology = detector.detect_ideology(text)
#         print(f"'{text}' -> Detected ideology: {ideology}")