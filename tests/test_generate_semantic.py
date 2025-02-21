import unittest
import numpy as np
import torch
from unittest.mock import patch, MagicMock
from typing import Union, Optional
from core.bark import (
    generate_semantic_tokens_from_text,
    SEMANTIC_VOCAB_SIZE,
    # SEMANTIC_PAD_TOKEN,
    # TEXT_PAD_TOKEN,
    # SEMANTIC_INFER_TOKEN,
    # _preprocess_texts,
    # trim_or_pad_array,
)


# Mock environment variables and model loading for isolation
class MockEnv:
    SUNO_USE_SMALL_MODELS = "True"


# Mock model and tokenizer classes
class MockGPT:
    def __init__(self):
        self.device = torch.device("cpu")

    def parameters(self):
        return [torch.tensor(0)]  # Dummy parameter for device detection

    def __call__(self, x, merge_context=True, use_cache=False, past_kv=None):
        # Simulate model output: predict next token as input shifted
        batch, seq_len = x.shape
        logits = torch.zeros((batch, seq_len, SEMANTIC_VOCAB_SIZE + 1))  # +1 for EOS
        for b in range(batch):
            for s in range(seq_len):
                token = x[b, s].item()
                if token < SEMANTIC_VOCAB_SIZE:
                    logits[b, s, token] = 1.0  # High probability for same token
                else:
                    logits[b, s, SEMANTIC_VOCAB_SIZE] = 1.0  # EOS for infer token
        return logits, None  # No kv_cache for simplicity


class MockBertTokenizer:
    def encode(self, text, add_special_tokens=False):
        # Simple mock: convert text to list of numbers (length = len(words))
        return [ord(c) % 100 for c in text.split()]  # Arbitrary tokenization

    def decode(self, tokens):
        return " ".join(str(t) for t in tokens)


class TestGenerateSemanticTokens(unittest.TestCase):
    def setUp(self):
        # Mock environment and model loading
        self.patcher_env = patch("core.bark.env", MockEnv())
        self.patcher_env.start()

        # Mock torch_models.get_model
        self.mock_model_info = MagicMock()
        self.mock_model_info.model = MockGPT()
        self.mock_model_info.preprocessor = MockBertTokenizer()
        self.patcher_model = patch(
            "core.bark.torch_models.get_model", return_value=self.mock_model_info
        )
        self.patcher_model.start()

    def tearDown(self):
        self.patcher_env.stop()
        self.patcher_model.stop()

    def test_basic_generation(self):
        """Test basic semantic token generation with default parameters."""
        text = "Hello world"
        output = generate_semantic_tokens_from_text(text, silent=True)

        # Check output is a tensor
        self.assertIsInstance(output, torch.Tensor)
        # Output should be 1D (sequence of tokens)
        self.assertEqual(len(output.shape), 1)
        # Tokens should be within semantic vocab range
        self.assertTrue(torch.all(output >= 0))
        self.assertTrue(torch.all(output < SEMANTIC_VOCAB_SIZE))

    def test_with_semantic_prompt(self):
        """Test generation with a semantic prompt."""
        text = "Test prompt"
        semantic_prompt = np.array([1, 2, 3])
        output = generate_semantic_tokens_from_text(
            text, semantic_prompt=semantic_prompt, silent=True
        )

        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(len(output.shape), 1)
        # Check if output length is reasonable (non-zero generation)
        self.assertGreater(len(output), 0)

    def test_empty_text(self):
        """Test that empty text raises an assertion error."""
        with self.assertRaises(AssertionError):
            generate_semantic_tokens_from_text("", silent=True)

    def test_max_duration_stop(self):
        """Test stopping based on max_gen_duration_second."""
        text = "Short text"
        output_short = generate_semantic_tokens_from_text(
            text, max_gen_duration_second=0.1, silent=True
        )
        output_long = generate_semantic_tokens_from_text(
            text, max_gen_duration_second=1.0, silent=True
        )

        # Shorter duration should produce fewer tokens
        self.assertLess(len(output_short), len(output_long))

    def test_early_stopping(self):
        """Test early stopping with EOS token simulation."""

        # Patch model to always return EOS after 5 tokens
        def mock_model_with_eos(x, **kwargs):
            batch, seq_len = x.shape
            logits = torch.zeros((batch, seq_len, SEMANTIC_VOCAB_SIZE + 1))
            if seq_len > 5:  # Trigger EOS after 5 steps
                logits[:, -1, SEMANTIC_VOCAB_SIZE] = 10.0  # High EOS logit
            else:
                logits[:, -1, 0] = 1.0  # Continue with token 0
            return logits, None

        with patch.object(self.mock_model_info, "model", MockGPT()) as mock_gpt:
            mock_gpt.__call__ = mock_model_with_eos
            output = generate_semantic_tokens_from_text(
                "Continue until EOS", allow_early_stop=True, silent=True
            )
            # Should stop after ~5 tokens due to EOS
            self.assertLessEqual(len(output), 6)

    def test_kv_caching(self):
        """Test that KV caching reduces input size after first step."""
        text = "KV caching test"
        with patch("core.bark._inference_bark") as mock_inference:
            generate_semantic_tokens_from_text(text, use_kv_caching=True, silent=True)
            # Check that inference was called with use_kv_caching=True
            mock_inference.assert_called_with(
                self.mock_model_info.model,
                unittest.mock.ANY,  # input_tensor
                temperature=0.7,
                top_k=None,
                top_p=None,
                silent=True,
                min_eos_p=0.2,
                max_gen_duration_s=None,
                allow_early_stop=True,
                use_kv_caching=True,
            )

    def test_top_k_filtering(self):
        """Test that top_k limits token diversity."""
        text = "Top k test"
        output_k10 = generate_semantic_tokens_from_text(
            text, top_k=10, temperature=1.0, silent=True
        )
        unique_tokens = len(torch.unique(output_k10))
        self.assertLessEqual(unique_tokens, 10)  # Should limit to top 10 tokens

    def test_input_tensor_shape(self):
        """Test that input tensor has correct shape [1, 513]."""
        text = "Shape test"
        with self.assertRaises(AssertionError):
            # Patch trim_or_pad_array to break shape
            with patch("core.bark.trim_or_pad_array", return_value=np.zeros(100)):
                generate_semantic_tokens_from_text(text, silent=True)


if __name__ == "__main__":
    unittest.main()
