import unittest

from utils import extract_explanation, extract_from_code_block

# NOTE to Ian - entirely written by Cursor


class TestExtractExplanation(unittest.TestCase):
    def test_extract_explanation_with_tags(self):
        """Test extracting explanation when tags are present"""
        text = "Here is some text <EXPLANATION>This is the explanation content</EXPLANATION> and more text"
        result = extract_explanation(text)
        self.assertEqual(result, "This is the explanation content")

    def test_extract_explanation_multiline(self):
        """Test extracting multiline explanation"""
        text = """Some text before
        <EXPLANATION>
        This is a multiline
        explanation with
        multiple lines
        </EXPLANATION>
        Some text after"""
        result = extract_explanation(text)
        expected = (
            "This is a multiline\n        explanation with\n        multiple lines"
        )
        self.assertEqual(result, expected)

    def test_extract_explanation_with_whitespace(self):
        """Test extracting explanation with leading/trailing whitespace"""
        text = "Text <EXPLANATION>  \n  Explanation with whitespace  \n  </EXPLANATION> more text"
        result = extract_explanation(text)
        self.assertEqual(result, "Explanation with whitespace")

    def test_extract_explanation_no_tags(self):
        """Test when no explanation tags are present"""
        text = "This text has no explanation tags"
        result = extract_explanation(text)
        self.assertEqual(result, "")

    def test_extract_explanation_empty_tags(self):
        """Test when explanation tags are empty"""
        text = "Text <EXPLANATION></EXPLANATION> more text"
        result = extract_explanation(text)
        self.assertEqual(result, "")

    def test_extract_explanation_multiple_tags(self):
        """Test when multiple explanation tags are present - should get first one"""
        text = "Text <EXPLANATION>First explanation</EXPLANATION> middle <EXPLANATION>Second explanation</EXPLANATION> end"
        result = extract_explanation(text)
        self.assertEqual(result, "First explanation")

    def test_extract_explanation_nested_content(self):
        """Test extracting explanation with nested content"""
        text = "Text <EXPLANATION>Explanation with <brackets> and other content</EXPLANATION> more"
        result = extract_explanation(text)
        self.assertEqual(result, "Explanation with <brackets> and other content")


class TestExtractFromCodeBlock(unittest.TestCase):
    def test_extract_standard_python_block(self):
        """Test extracting code from standard ```python``` block"""
        text = "Here is some code:\n```python\ndef transform(x):\n    return x\n```\nMore text"
        result = extract_from_code_block(text)
        self.assertEqual(result, "def transform(x):\n    return x")

    def test_extract_no_language_specified(self):
        """Test extracting code from ``` block without language"""
        text = "Code:\n```\ndef transform(x):\n    return x\n```"
        result = extract_from_code_block(text)
        self.assertEqual(result, "def transform(x):\n    return x")

    def test_extract_with_no_whitespace(self):
        """Test extracting code that starts immediately after backticks"""
        text = "```python\ndef transform(x):\n    return x```"
        result = extract_from_code_block(text)
        self.assertIsNotNone(result)
        self.assertIn("def transform", result)

    def test_extract_plain_def_transform(self):
        """Test extracting code that's not in markdown blocks but starts with def transform"""
        text = "Here's the solution:\ndef transform(grid):\n    return grid[::-1]\n\nThat should work!"
        result = extract_from_code_block(text)
        self.assertIsNotNone(result)
        self.assertIn("def transform(grid)", result)

    def test_extract_no_code_found(self):
        """Test when no code block is present"""
        text = "This is just plain text with no code blocks"
        result = extract_from_code_block(text)
        self.assertIsNone(result)

    def test_extract_empty_code_block(self):
        """Test when code block is empty"""
        text = "Code:\n```python\n```"
        result = extract_from_code_block(text)
        # Should return None for empty blocks (after strip)
        self.assertIsNone(result)

    def test_extract_multiple_blocks(self):
        """Test extracting first code block when multiple are present"""
        text = "```python\ndef first():\n    pass\n```\nText\n```python\ndef second():\n    pass\n```"
        result = extract_from_code_block(text)
        self.assertIn("first", result)
        self.assertNotIn("second", result)


if __name__ == "__main__":
    unittest.main()
