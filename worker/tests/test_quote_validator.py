import unittest

from app import validate_quote


class QuoteValidatorTests(unittest.TestCase):
    def test_exact_span(self):
        text = "abc def ghi"
        quote = "def"
        result = validate_quote(text, quote, 4, 7)
        self.assertEqual(result, (4, 7))

    def test_fallback_search(self):
        text = "hello world"
        quote = "world"
        result = validate_quote(text, quote, 0, 5)
        self.assertEqual(result, (6, 11))

    def test_missing_quote(self):
        text = "hello"
        quote = "absent"
        result = validate_quote(text, quote, 0, 6)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
