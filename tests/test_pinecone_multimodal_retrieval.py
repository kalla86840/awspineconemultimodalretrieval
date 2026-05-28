import sys
import types
import unittest
from unittest.mock import patch

from rag_endpoint import app


class FakeResponses:
    def create(self, **kwargs):
        self.kwargs = kwargs
        return type("Response", (), {"output_text": "A server rack image with cloud infrastructure equipment."})()


class FakeOpenAIClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.responses = FakeResponses()


class FakeIndex:
    def query(self, **kwargs):
        self.kwargs = kwargs
        return {
            "matches": [
                {
                    "id": "record-1",
                    "score": 0.88,
                    "metadata": {
                        "title": "EGIO edge cloud transcript",
                        "content": "Earnings call commentary about edge cloud infrastructure.",
                    },
                }
            ]
        }


class PineconeMultimodalRetrievalTests(unittest.TestCase):
    def test_normalize_image_inputs_accepts_urls_and_base64(self):
        images = app.normalize_image_inputs(
            {
                "image_url": "https://example.com/server.jpg",
                "images": [{"base64": "abc123", "mime_type": "image/png"}],
            }
        )

        self.assertEqual(images[0]["image_url"], "https://example.com/server.jpg")
        self.assertEqual(images[1]["image_url"], "data:image/png;base64,abc123")

    def test_multimodal_retrieval_fuses_caption_and_queries_pinecone(self):
        fake_openai_module = types.SimpleNamespace(OpenAI=FakeOpenAIClient)
        fake_index = FakeIndex()
        payload = {
            "mode": "multimodal_retrieval",
            "question": "Find related earnings context.",
            "image_url": "https://example.com/server.jpg",
            "metadata": {"ticker": "EGIO"},
            "namespace": "news",
        }

        with patch.dict(sys.modules, {"openai": fake_openai_module}):
            with patch.object(app, "get_openai_api_key", return_value="test-key"):
                with patch.object(app, "get_pinecone_index", return_value=fake_index):
                    with patch.object(app, "embed_texts", return_value=[[0.1, 0.2, 0.3]]):
                        result = app.multimodal_retrieval(payload, documents=[], top_k=3)

        self.assertEqual(result["mode"], "multimodal_retrieval")
        self.assertEqual(result["retrieval_source"], "pinecone")
        self.assertEqual(result["image_count"], 1)
        self.assertIn("server rack image", result["fused_query"])
        self.assertEqual(fake_index.kwargs["namespace"], "news")
        self.assertEqual(result["retrieved_context"][0]["id"], "record-1")


if __name__ == "__main__":
    unittest.main()
