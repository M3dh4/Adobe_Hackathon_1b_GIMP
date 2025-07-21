import re

def detect_headings_by_language(lines):
    """
    Return a set of lines likely to be headings based on language cues.
    """
    heading_patterns = [
        r"^introduction$", r"^conclusion$", r"^summary$", r"^references$", r"^appendix$",
        r"^table of contents$", r"^background$", r"^results$", r"^discussion$", r"^abstract$",
        r"^methodology$", r"^goals$", r"^objectives$", r"^requirements$", r"^evaluation$",
        r"^approach$", r"^timeline$", r"^milestones$", r"^structure$", r"^outcomes$"
    ]
    heading_regex = re.compile("|".join(heading_patterns), re.IGNORECASE)
    return set(line for line in lines if heading_regex.match(line.strip())) 