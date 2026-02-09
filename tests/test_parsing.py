"""
Tests for extract_json_from_response() in gemini_webapi.utils.parsing.

Validates the upstream length-marker based parser which uses UTF-16 code unit
counts to determine chunk boundaries in Google RPC streaming responses.

Format:
    )]}'              <- XSSI protection header
    <utf16_length>    <- UTF-16 code unit count of content after marker digits
    <json_payload>    <- JSON chunk (may span multiple lines)
"""

import unittest

from gemini_webapi.utils.parsing import extract_json_from_response


def _utf16_len(s: str) -> int:
    """Count UTF-16 code units in a string."""
    return sum(2 if ord(c) > 0xFFFF else 1 for c in s)


def _build_response(json_content: str, xssi: bool = True) -> str:
    """Build a Google RPC response with correct UTF-16 length marker.

    The marker counts UTF-16 units starting from the character immediately
    after the marker's digits (the newline separator between marker and content).
    """
    after_digits = "\n" + json_content
    marker = _utf16_len(after_digits)
    header = ")]}'\n" if xssi else ""
    return f"{header}{marker}\n{json_content}"


class TestExtractJsonFromResponse(unittest.TestCase):
    """Test suite for extract_json_from_response() function."""

    # ---- Multi-line JSON chunk tests ----

    def test_multiline_json_chunk_basic(self):
        """Multi-line JSON chunk should be parsed correctly via length marker."""
        json_content = (
            '[["wrb.fr","BardGeneratorService","GetReply",\n'
            '"some data here",\n'
            '"more data"\n'
            ']]'
        )
        response = _build_response(json_content)
        result = extract_json_from_response(response)
        self.assertIsInstance(result, list)
        self.assertEqual(result[0][0], "wrb.fr")
        self.assertEqual(result[0][1], "BardGeneratorService")

    def test_multiline_json_chunk_with_nested_arrays(self):
        """Multi-line JSON with nested structures should parse correctly."""
        json_content = (
            '[["wrb.fr","BardGeneratorService","GetReply",[\n'
            '    ["nested", "array", "data"],\n'
            '    ["another", "nested", "item"]\n'
            '],null]]'
        )
        response = _build_response(json_content)
        result = extract_json_from_response(response)
        self.assertIsInstance(result, list)
        self.assertEqual(result[0][0], "wrb.fr")
        self.assertIsInstance(result[0][3], list)
        self.assertEqual(result[0][3][0], ["nested", "array", "data"])

    def test_multiline_json_with_escaped_newlines_in_strings(self):
        """JSON containing escape sequences within string values should work."""
        json_content = (
            '[["wrb.fr","response",\n'
            '"This is a string\\nwith escaped newlines\\ninside it"\n'
            ']]'
        )
        response = _build_response(json_content)
        result = extract_json_from_response(response)
        self.assertIsInstance(result, list)
        self.assertEqual(result[0][0], "wrb.fr")
        # After JSON parsing, \n becomes actual newline characters
        self.assertIn("\n", result[0][2])

    def test_multiline_json_chunk_realistic_response(self):
        """Realistic multi-line response structure from Google API."""
        # Newlines placed between outer JSON elements (valid whitespace),
        # not inside JSON string values (which would be invalid JSON).
        json_content = (
            '[["wrb.fr","BardGeneratorService","GetReply",'
            '"[[[\\"Hello! How can I help you today?\\",'
            'null,null,null,null,null,null,null,null,null,null,null,null,null,null,'
            'null,null,null,null,null,null,null,null,null,null,null,null,null,null,'
            'null,null,null,null,null,null,null,null,null,[\\"en\\"],null,null,null,'
            'null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,'
            'null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,'
            'null,null,null,null,null,null,[\\"c_abc123\\",\\"r_xyz789\\",\\"rc_def456\\"],'
            'null,null,null,null,null,null,null,null,null,null,null,null,'
            'null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null'
            ']]]",\nnull,\nnull,\nnull,\n"generic"]]'
        )
        response = _build_response(json_content)
        result = extract_json_from_response(response)
        self.assertIsInstance(result, list)
        self.assertEqual(result[0][0], "wrb.fr")
        self.assertEqual(result[0][1], "BardGeneratorService")
        self.assertEqual(result[0][2], "GetReply")

    # ---- Backward compatibility tests ----

    def test_single_line_json_chunk(self):
        """Single-line JSON chunks should work (backward compatibility)."""
        json_content = '[["wrb.fr","BardGeneratorService","GetReply","data",null]]'
        response = _build_response(json_content)
        result = extract_json_from_response(response)
        self.assertIsInstance(result, list)
        self.assertEqual(result[0][0], "wrb.fr")

    def test_single_line_simple_array(self):
        """Simple flat JSON array should parse correctly."""
        json_content = '["simple", "array", 123]'
        response = _build_response(json_content)
        result = extract_json_from_response(response)
        # Flat list gets extended directly
        self.assertEqual(result, ["simple", "array", 123])

    def test_single_line_without_xssi_header(self):
        """Response without XSSI header but with length marker should work."""
        json_content = '[["wrb.fr","BardGeneratorService","GetReply","data",null]]'
        response = _build_response(json_content, xssi=False)
        result = extract_json_from_response(response)
        self.assertIsInstance(result, list)
        self.assertEqual(result[0][0], "wrb.fr")

    def test_json_only_no_markers(self):
        """Plain JSON without any markers should still parse."""
        response = '[["wrb.fr","BardGeneratorService","GetReply"]]'
        result = extract_json_from_response(response)
        self.assertIsInstance(result, list)
        self.assertEqual(result[0][0], "wrb.fr")

    # ---- Length marker format tests ----

    def test_length_marker_before_json_chunk(self):
        """Length marker should correctly determine chunk boundaries."""
        json_content = '[["wrb.fr","test","data"]]'
        response = _build_response(json_content)
        result = extract_json_from_response(response)
        self.assertIsInstance(result, list)
        self.assertEqual(result[0][0], "wrb.fr")
        self.assertEqual(result[0][1], "test")

    def test_multiple_chunks(self):
        """Multiple JSON chunks should all be parsed and combined."""
        chunk1 = '[["wrb.fr","first","chunk"]]'
        chunk2 = '[["second","chunk","here"]]'
        marker1 = _utf16_len("\n" + chunk1)
        marker2 = _utf16_len("\n" + chunk2)
        response = f")]}}\'\n{marker1}\n{chunk1}\n{marker2}\n{chunk2}"
        result = extract_json_from_response(response)
        self.assertIsInstance(result, list)
        # Both chunks are extended into the result
        self.assertEqual(result[0][1], "first")
        self.assertEqual(result[1][1], "chunk")

    def test_length_marker_with_multiline_chunk(self):
        """Length marker should correctly bound a multi-line JSON chunk."""
        json_content = '[["wrb.fr",\n"multiline",\n"chunk"]]'
        response = _build_response(json_content)
        result = extract_json_from_response(response)
        self.assertIsInstance(result, list)
        self.assertEqual(result[0][0], "wrb.fr")
        self.assertEqual(result[0][1], "multiline")
        self.assertEqual(result[0][2], "chunk")

    # ---- Edge cases and error handling ----

    def test_empty_string_raises_value_error(self):
        """Empty input should raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            extract_json_from_response("")
        self.assertIn("Could not find", str(ctx.exception))

    def test_non_string_input_raises_type_error(self):
        """Non-string input should raise TypeError."""
        with self.assertRaises(TypeError):
            extract_json_from_response(None)
        with self.assertRaises(TypeError):
            extract_json_from_response(123)
        with self.assertRaises(TypeError):
            extract_json_from_response(["list", "input"])

    def test_no_json_in_response_raises_value_error(self):
        """Response with no valid JSON should raise ValueError."""
        response = ")]}'\nThis is not JSON\nJust plain text"
        with self.assertRaises(ValueError) as ctx:
            extract_json_from_response(response)
        self.assertIn("Could not find", str(ctx.exception))

    def test_malformed_json_raises_value_error(self):
        """Malformed JSON should raise ValueError."""
        response = ')]}\'\\n50\\n[["unclosed", "array"'
        with self.assertRaises(ValueError):
            extract_json_from_response(response)

    def test_xssi_header_only_raises_value_error(self):
        """Response with only XSSI header should raise ValueError."""
        response = ")]}'"
        with self.assertRaises(ValueError):
            extract_json_from_response(response)

    def test_whitespace_around_json(self):
        """Trailing whitespace should be handled gracefully."""
        json_content = '[["wrb.fr","test","data"]]'
        response = _build_response(json_content) + "\n   "
        result = extract_json_from_response(response)
        self.assertIsInstance(result, list)
        self.assertEqual(result[0][0], "wrb.fr")

    def test_json_object_instead_of_array(self):
        """JSON objects should be parseable (wrapped in list by upstream)."""
        json_content = '{"key": "value", "number": 42}'
        response = _build_response(json_content)
        result = extract_json_from_response(response)
        self.assertIsInstance(result, list)
        self.assertEqual(result[0]["key"], "value")
        self.assertEqual(result[0]["number"], 42)


class TestExtractJsonMultilineEdgeCases(unittest.TestCase):
    """Additional edge cases for multi-line JSON parsing."""

    def test_json_with_unicode_multiline(self):
        """Multi-line JSON containing unicode characters should parse correctly."""
        json_content = '[["wrb.fr",\n"Hello",\n"unicode characters"]]'
        response = _build_response(json_content)
        result = extract_json_from_response(response)
        self.assertIsInstance(result, list)
        self.assertEqual(result[0][0], "wrb.fr")

    def test_deeply_nested_multiline_structure(self):
        """Deeply nested multi-line JSON structures should parse correctly."""
        json_content = (
            '[["wrb.fr","service",[\n'
            '    [\n'
            '        [\n'
            '            "deeply",\n'
            '            "nested",\n'
            '            [\n'
            '                "structure"\n'
            '            ]\n'
            '        ]\n'
            '    ]\n'
            ']]]'
        )
        response = _build_response(json_content)
        result = extract_json_from_response(response)
        self.assertIsInstance(result, list)
        self.assertEqual(result[0][0], "wrb.fr")
        self.assertEqual(result[0][1], "service")
        self.assertEqual(result[0][2][0][0][0], "deeply")

    def test_json_with_null_values_multiline(self):
        """Multi-line JSON with null values should parse correctly."""
        json_content = '[["wrb.fr",\nnull,\n"data",\nnull,\nnull]]'
        response = _build_response(json_content)
        result = extract_json_from_response(response)
        self.assertIsInstance(result, list)
        self.assertEqual(result[0][0], "wrb.fr")
        self.assertIsNone(result[0][1])
        self.assertEqual(result[0][2], "data")
        self.assertIsNone(result[0][3])

    def test_json_with_boolean_values_multiline(self):
        """Multi-line JSON with boolean values should parse correctly."""
        json_content = '[["wrb.fr",\ntrue,\nfalse,\n"string",\ntrue]]'
        response = _build_response(json_content)
        result = extract_json_from_response(response)
        self.assertIsInstance(result, list)
        self.assertEqual(result[0][0], "wrb.fr")
        self.assertTrue(result[0][1])
        self.assertFalse(result[0][2])

    def test_json_with_numeric_values_multiline(self):
        """Multi-line JSON with various numeric types should parse correctly."""
        json_content = '[["wrb.fr",\n42,\n3.14159,\n-100,\n1.5e10]]'
        response = _build_response(json_content)
        result = extract_json_from_response(response)
        self.assertIsInstance(result, list)
        self.assertEqual(result[0][0], "wrb.fr")
        self.assertEqual(result[0][1], 42)
        self.assertAlmostEqual(result[0][2], 3.14159)
        self.assertEqual(result[0][3], -100)
        self.assertEqual(result[0][4], 1.5e10)


if __name__ == "__main__":
    unittest.main()
