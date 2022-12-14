from models import models
import torch

def test_models():
    for modelname, model_type in models.items():
        sample_batch = torch.rand(5,3,224,224)
        model = model_type(input_size = (3, 224, 224),
                          hidden_size = 20)
        out, mu, log_var = model(sample_batch)
        assert out.size() == torch.Size((5, 3, 224, 224)), f"{modelname}: Wrong Reconstruction Image Dimension"
        assert mu.size() == torch.Size((5, 20)), f"{modelname}: Wrong Latent Distribution Mean Dimension"
        assert log_var.size() == torch.Size((5, 20)), f"{modelname}: Wrong Latent Distribution Variance Dimension"
        