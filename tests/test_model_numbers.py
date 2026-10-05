"""model_numbers(): real model codes only; ASINs, long hex ids and unmixed tokens are excluded."""
import unittest

import support  # noqa: F401  (network guard)

import tools


class ModelNumbersTest(unittest.TestCase):
    def test_real_model_codes_kept(self):
        for code in ("fd0070tu", "fy5007tu", "82RK0085IN", "RT38HG5A42S8HL", "ep1180tu", "x1504zanj522ws"):
            self.assertEqual(tools.model_numbers(code), {code.lower()}, code)

    def test_amazon_asins_excluded(self):
        # ids seen next to carousel prices in live run 2
        for asin in ("b0d2y1bldt", "B0G2BHDDB8", "b0cy2plq8n", "b0f671gg5m", "B0DCG26YC5"):
            self.assertEqual(tools.model_numbers(asin), set(), asin)

    def test_long_hex_ids_excluded(self):
        for hex_id in ("ec1eb99eca6c", "b1e596c09d6d", "3f2a9c1b7e4d5a60"):
            self.assertEqual(tools.model_numbers(hex_id), set(), hex_id)

    def test_8_char_hex_ids_excluded(self):
        # ids seen next to Samsung page prices in live run 3
        for hex_id in ("dfa6080d", "edae522d", "DFA6080D"):
            self.assertEqual(tools.model_numbers(hex_id), set(), hex_id)

    def test_hp_models_from_live_runs_still_accepted(self):
        for code in ("fd0022tu", "fd0577tu", "fy5007tu", "FD0022TU"):
            self.assertEqual(tools.model_numbers(f"HP 15 {code} Laptop"), {code.lower()}, code)

    def test_unmixed_tokens_excluded(self):
        for token in ("abcdefghij", "1234567890", "thinkpadx", "a1234567"):
            self.assertEqual(tools.model_numbers(token), set(), token)

    def test_cpu_gpu_ram_tokens_excluded(self):
        self.assertEqual(tools.model_numbers("Intel Core i5-1235U RTX4050 16GB Ryzen 7520U snapdragon8gen3"), set())

    def test_mixed_text(self):
        text = "HP 15s fy5007TU /dp/B0CJBP38HR image ec1eb99eca6c"
        self.assertEqual(tools.model_numbers(text), {"fy5007tu"})


if __name__ == "__main__":
    unittest.main()
