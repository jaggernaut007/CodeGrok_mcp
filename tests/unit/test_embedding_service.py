import pytest
from unittest.mock import MagicMock, patch
from codegrok_mcp.indexing.embedding_service import (
    EmbeddingService,
    get_embedding_service,
    reset_embedding_service,
)


class TestEmbeddingService:
    def teardown_method(self):
        reset_embedding_service()

    def test_embed_empty_batch(self):
        service = EmbeddingService(model_name="all-MiniLM-L6-v2")
        # Mock _model to avoid downloading/loading
        service._model = MagicMock()
        service._model_loaded = True

        result = service.embed_batch([])
        assert result == []

    def test_cache_statistics(self):
        service = EmbeddingService(model_name="all-MiniLM-L6-v2")

        # Mock dependencies
        with patch("codegrok_mcp.indexing.embedding_service._sentence_transformers") as mock_st:
            mock_model = MagicMock()
            # mock encode return value
            import numpy as np

            mock_model.encode.return_value = np.array([[0.1, 0.2]])
            mock_model.get_sentence_embedding_dimension.return_value = 2

            # Manually load mock model
            service._model = mock_model
            service.config["dimensions"] = 2
            service._model_loaded = True

            service.embed("test")
            service.embed("test")  # Cache hit

            stats = service.get_cache_stats()

            assert service.stats["cache_hits"] == 1
            assert service.stats["cache_misses"] == 1
            assert stats["hits"] == 1

    def test_unload_model(self):
        service = EmbeddingService(model_name="all-MiniLM-L6-v2")
        service._model = MagicMock()
        service._model_loaded = True

        service.unload()
        assert service._model is None
        assert service._model_loaded is False

    def test_singleton_pattern(self):
        reset_embedding_service()
        # Mock dependencies to avoid import errors if not installed
        with patch("codegrok_mcp.indexing.embedding_service._import_dependencies"):
            s1 = get_embedding_service()
            s2 = get_embedding_service()
            assert s1 is s2
