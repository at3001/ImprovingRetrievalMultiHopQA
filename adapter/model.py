import torch
from torch import nn

class Adapter(nn.Module):
  def __init__(self, in_dimension, latent_dim):
    super().__init__()
    self.dimension = in_dimension
    self.latent = latent_dim

    self.chunk_project1 = nn.Linear(in_dimension, latent_dim, bias=False)
    self.chunk_project2 = nn.Linear(latent_dim, in_dimension, bias=False)

    self.query_project1 = nn.Linear(in_dimension, latent_dim, bias=False)
    self.query_project2 = nn.Linear(latent_dim, in_dimension, bias=False)

    self.init_weights()

  def init_weights(self):
    self.chunk_project2.weight.data.copy_(torch.zeros(self.dimension, self.latent))
    self.query_project2.weight.data.copy_(torch.zeros(self.dimension, self.latent))

  def forward(self, chunk_batch, query_batch):
    # Both batches should be N * D
    chunk_adapter = self.chunk_project2(nn.functional.dropout(self.chunk_project1(chunk_batch), p=0.5))
    query_adapter = self.query_project2(nn.functional.dropout(self.query_project1(query_batch), p=0.5))
    new_chunk = chunk_batch + chunk_adapter
    new_query = query_batch + query_adapter
    cosine_similarity = nn.functional.cosine_similarity(new_chunk, new_query)
    return new_chunk, new_query, cosine_similarity

class Projecter(nn.Module):
  def __init__(self, in_dimension):
    super().__init__()
    self.dimension = in_dimension
    
    self.chunk_project = nn.Linear(in_dimension, in_dimension)
    self.query_project = nn.Linear(in_dimension, in_dimension)
    
    self.init_weights()
    
  def init_weights(self):
    self.chunk_project.weight.data.copy_(torch.eye(self.dimension) + torch.sqrt(torch.tensor([0.1]))*torch.randn(self.dimension, self.dimension))
    self.query_project.weight.data.copy_(torch.eye(self.dimension) + torch.sqrt(torch.tensor([0.1]))*torch.randn(self.dimension, self.dimension))
    self.chunk_project.bias.data.copy_(torch.zeros(self.dimension) + torch.sqrt(torch.tensor([0.1]))*torch.randn(self.dimension))
    self.query_project.bias.data.copy_(torch.zeros(self.dimension) + torch.sqrt(torch.tensor([0.1]))*torch.randn(self.dimension))


  def forward(self, chunk_batch, query_batch):
    # Both batches should be N * D
    chunk_adapter = self.chunk_project(nn.functional.dropout(chunk_batch, 0.4))
    query_adapter = self.query_project(nn.functional.dropout(query_batch, 0.4))
    chunk_adapter = nn.functional.normalize(chunk_adapter, dim=1)
    query_adapter = nn.functional.normalize(query_adapter, dim=1)
    metric = nn.functional.cosine_similarity(chunk_adapter, query_adapter)
    return chunk_adapter, query_adapter, metric

class Concatenater(nn.Module):
  def __init__(self, in_dimension):
    super().__init__()
    self.dimension = in_dimension
    
    self.chunk_project = nn.Linear(in_dimension, 32)
    self.query_project = nn.Linear(in_dimension, 32)

  def forward(self, chunk_batch, query_batch):
    # Both batches should be N * D
    chunk_adapter = self.chunk_project(nn.functional.dropout(chunk_batch, 0.4))
    query_adapter = self.query_project(nn.functional.dropout(query_batch, 0.4))
    chunk_concatenated = torch.cat((chunk_adapter, chunk_batch), dim=1)
    query_concatenated = torch.cat((query_adapter, query_batch), dim=1)
    metric = nn.functional.cosine_similarity(chunk_adapter, query_adapter)
    return chunk_adapter, query_adapter, metric