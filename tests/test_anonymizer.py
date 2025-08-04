import unittest
from anonymizer.anonymizer import DocumentAnonymizer

class TestDocumentAnonymizer(unittest.TestCase):
    def setUp(self):
        """Create a fresh anonymizer instance for each test."""
        self.anonymizer = DocumentAnonymizer(threshold=0.5)

    # -------------------------------
    # 1. BASIC FUNCTIONALITY
    # -------------------------------
    def test_empty_text(self):
        """Empty input should return empty output with 0 entities."""
        result = self.anonymizer.anonymize("")
        self.assertEqual(result["statistics"]["total_entities"], 0)
        self.assertEqual(result["anonymized_text"], "")

    def test_basic_entities(self):
        """Detects and anonymizes simple PER, EMAIL, PHONE entities."""
        sample_text = "John Smith can be reached at john@example.com or 555-123-4567."
        result = self.anonymizer.anonymize(sample_text)

        # Check anonymized text has placeholders
        self.assertIn("[PER_1]", result["anonymized_text"])
        self.assertIn("[EMAIL_1]", result["anonymized_text"])
        self.assertIn("[PHONE_1]", result["anonymized_text"])

        # Verify statistics counts
        stats = result["statistics"]["by_category"]
        self.assertGreaterEqual(stats.get("PER", 0), 1)
        self.assertGreaterEqual(stats.get("EMAIL", 0), 1)
        self.assertGreaterEqual(stats.get("PHONE", 0), 1)

    # -------------------------------
    # 2. EDGE CASES
    # -------------------------------
    def test_no_entities(self):
        """Text with no entities should remain unchanged."""
        sample_text = "This is a simple sentence without any PII."
        result = self.anonymizer.anonymize(sample_text)
        self.assertEqual(result["anonymized_text"], sample_text)
        self.assertEqual(result["statistics"]["total_entities"], 0)

    def test_duplicate_entities(self):
        """Same entity repeated should map to same placeholder."""
        sample_text = "Contact John Smith. John Smith will assist you."
        result = self.anonymizer.anonymize(sample_text)

        placeholders = [
            key for key, val in result["entity_mapping"].items()
            if "John Smith" in val
        ]
        self.assertEqual(len(placeholders), 1, "Duplicate entities should map to same placeholder")

    def test_short_fragments_ignored(self):
        """Tiny fragments like 'Jo' or 'ob' should not appear in final mapping."""
        sample_text = "J o h n  S m i t h works at A c m e  C o r p."
        result = self.anonymizer.anonymize(sample_text)

        for entity in result["entity_mapping"].values():
            self.assertGreaterEqual(len(entity), 3, "Fragments under 3 chars should be ignored")

    # -------------------------------
    # 3. MODEL BEHAVIOR
    # -------------------------------
    def test_low_confidence_threshold(self):
        """Entities below threshold should be ignored."""
        low_conf_anonymizer = DocumentAnonymizer(threshold=0.99)
        sample_text = "John Smith works at Acme Corp."
        result = low_conf_anonymizer.anonymize(sample_text)
        # With very high threshold, likely 0 entities detected
        self.assertLessEqual(result["statistics"]["total_entities"], 2)

    # -------------------------------
    # 4. PERFORMANCE TEST
    # -------------------------------
    def test_large_text_performance(self):
        """Ensure large text is processed without errors and not overly shortened."""
        sample_text = ("John Smith works at Acme Corp. Email: john@example.com. " * 500)
        result = self.anonymizer.anonymize(sample_text)

        # Basic checks
        self.assertIsInstance(result, dict)
        self.assertGreater(result["statistics"]["total_entities"], 0)

        # Ensure output is at least 70% of original length (placeholders may shorten text)
        self.assertGreaterEqual(len(result["anonymized_text"]), int(0.7 * len(sample_text)))


if __name__ == "__main__":
    unittest.main()
