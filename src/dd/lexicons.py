"""Psycholinguistic lexicons. Starting points: validate/extend against literature
(e.g. Al-Mosaiwi & Johnstone 2018 for absolutist words; LIWC-style resources).

Caveat for Polish: it is a pro-drop language, so explicit 'ja' is rarer than 'I' in
English and first-person is also marked by verb endings (-em, -am, -ę). Treat the
PL pronoun feature as weaker and test it separately (ablation per language).
"""

FIRST_PERSON_SINGULAR = {
    "en": {"i", "me", "my", "mine", "myself", "im", "ive", "id"},
    "pl": {
        "ja", "mnie", "mi", "mną", "mój", "moja", "moje", "mojego", "mojej", "moim",
        "moich", "moją", "moimi", "mym", "swój", "siebie", "sobie", "sobą",
    },
}

ABSOLUTIST = {
    "en": {
        "always", "never", "nothing", "nobody", "everything", "everyone", "completely",
        "totally", "absolutely", "entire", "entirely", "constantly", "forever", "every",
        "all", "whole", "definitely", "perfect", "ever",
    },
    "pl": {
        "zawsze", "nigdy", "nic", "nikt", "nikogo", "niczego", "wszystko", "wszyscy",
        "całkowicie", "kompletnie", "absolutnie", "ciągle", "wiecznie", "żaden", "żadna",
        "żadne", "wcale", "całkiem", "totalnie", "każdy", "cały",
    },
}
