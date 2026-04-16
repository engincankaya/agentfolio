import json
import unittest

from src.api.streaming import AnswerStreamExtractor, ndjson_event, normalize_suggestions


class StreamingTests(unittest.TestCase):
    def test_answer_extractor_reads_single_chunk(self):
        extractor = AnswerStreamExtractor()

        delta = extractor.feed('{"answer":"Merhaba","suggestions":[]}')

        self.assertEqual(delta, "Merhaba")
        self.assertTrue(extractor.done)

    def test_answer_extractor_reads_split_chunks(self):
        extractor = AnswerStreamExtractor()
        chunks = ['{"ans', 'wer"', ':"Eng', "incan", ' kimdir?","suggestions":["x"]}']

        output = "".join(extractor.feed(chunk) for chunk in chunks)

        self.assertEqual(output, "Engincan kimdir?")
        self.assertTrue(extractor.done)

    def test_answer_extractor_decodes_escaped_characters(self):
        extractor = AnswerStreamExtractor()

        output = extractor.feed(r'{"answer":"Satır 1\n\"Merhaba\" \u0130","suggestions":[]}')

        self.assertEqual(output, 'Satır 1\n"Merhaba" İ')

    def test_answer_extractor_does_not_emit_suggestions(self):
        extractor = AnswerStreamExtractor()

        output = extractor.feed('{"answer":"Cevap","suggestions":["Bunu göster"]}')
        output += extractor.feed('{"answer":"Yanlış alan"}')

        self.assertEqual(output, "Cevap")

    def test_ndjson_event_outputs_parseable_line(self):
        line = ndjson_event("answer_delta", content="Merhaba")

        self.assertTrue(line.endswith("\n"))
        self.assertEqual(json.loads(line), {"type": "answer_delta", "content": "Merhaba"})

    def test_normalize_suggestions_limits_and_trims(self):
        suggestions = normalize_suggestions([" A ", "", "B", "C", "D"])

        self.assertEqual(suggestions, ["A", "B", "C"])

    def test_normalize_suggestions_rejects_non_lists(self):
        self.assertEqual(normalize_suggestions({"0": "A"}), [])


if __name__ == "__main__":
    unittest.main()
