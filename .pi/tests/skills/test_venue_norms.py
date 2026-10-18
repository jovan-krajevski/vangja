import re
import unittest

from helpers import SKILLS

SKILL_DIR = SKILLS / "venue-norms-and-reviewing"
GRADES = (SKILL_DIR / "references" / "grades.md").read_text()
VENUES = (SKILL_DIR / "references" / "venues.md").read_text()
FORM = (SKILL_DIR / "templates" / "review_form.md").read_text()
SKILL_MD = (SKILL_DIR / "SKILL.md").read_text()

LADDER = ["strong accept", "accept", "weak accept", "borderline",
          "weak reject", "reject", "strong reject"]


class TestGradeLadder(unittest.TestCase):
    def test_ladder_order(self):
        positions = [GRADES.index(f"`{grade}`") for grade in LADDER]
        self.assertEqual(positions, sorted(positions), positions)

    def test_blocking_rules_present(self):
        self.assertIn("confirmed S0", GRADES)
        self.assertIn("cap `weak reject`", GRADES)
        self.assertIn("cap `borderline`", GRADES)
        self.assertIn("recall < 0.6", GRADES)

    def test_journal_mapping(self):
        self.assertIn("major revision", GRADES)
        self.assertIn("minor revision", GRADES)

    def test_skill_states_the_same_caps(self):
        self.assertIn("weak reject", SKILL_MD)
        self.assertIn("borderline", SKILL_MD)


class TestVenues(unittest.TestCase):
    def test_four_target_venues(self):
        for venue in ("Journal of Forecasting", "International Journal of Forecasting",
                      "Machine Learning", "Journal of Statistical Software"):
            self.assertIn(venue, VENUES, venue)

    def test_desk_reject_triggers_documented(self):
        self.assertIn("Desk-reject triggers", VENUES)
        self.assertIn("test-horizon selection", VENUES)
        self.assertIn("F-14", VENUES)

    def test_review_form_fields(self):
        for field in ("pr_id", "head_sha", "decision", "grade", "Rubric",
                      "Blocking-rule trace", "Required changes", "Residual risk"):
            self.assertIn(field, FORM, field)
        self.assertIn("author != reviewer", FORM)

    def test_venue_file_has_no_unverified_numbers(self):
        # A venue description must not smuggle in impact-factor claims.
        numbers = re.findall(r"\b[0-9]+\.[0-9]+\b", VENUES)
        self.assertEqual(numbers, [], numbers)


if __name__ == "__main__":
    unittest.main()
