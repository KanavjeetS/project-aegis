"""
Tests for AEGIS VLM and API
"""

import pytest
import torch
import torch.nn as nn
from fastapi.testclient import TestClient
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.vlm import AEGISModel
from models.vjepa import VJEPAModel
from api.api_server import app


# --- VLM Unit Tests ---

class TestAEGISModel:
    @pytest.fixture
    def model(self):
        return AEGISModel(
            vjepa_checkpoint=None,  # No checkpoint for testing
            use_quantized_llm=False  # CPU compatible
        )

    def test_initialization(self, model):
        assert isinstance(model.vjepa, VJEPAModel)
        assert isinstance(model.qformer, nn.Module)
    
    def test_forward_stage1(self, model):
        # Stage 1: V-JEPA + Q-Former (no LLM)
        video = torch.randn(2, 16, 3, 224, 224)
        text_input = ["test caption", "another caption"]
        
        # Mock V-JEPA output
        model.vjepa.extract_embeddings = lambda x: torch.randn(x.size(0), 16, 768)
        
        # Mock Q-Former output
        loss = model(video, text_input, stage=1)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad


# --- API Integration Tests ---

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "Project A.E.G.I.S."

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert "model_loaded" in response.json()

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "BLEU-4" in response.json()["metrics"]

def test_prediction_endpoint_no_file():
    response = client.post("/predict")
    # 422 Unprocessable Entity (missing file)
    assert response.status_code == 422

# Note: Full prediction test requires mocking the model for CPU/CI/CD
