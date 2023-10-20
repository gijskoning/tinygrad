import torch
import torch.nn.functional as F


def negative_cosine_similarity(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
  """
  Overview:
      consistency loss function: the negative cosine similarity.
  Arguments:
      - x1 (:obj:`torch.Tensor`): shape (batch_size, dim), e.g. (256, 512)
      - x2 (:obj:`torch.Tensor`): shape (batch_size, dim), e.g. (256, 512)
  Returns:
      (x1 * x2).sum(dim=1) is the cosine similarity between vector x1 and x2.
      The cosine similarity always belongs to the interval [-1, 1].
      For example, two proportional vectors have a cosine similarity of 1,
      two orthogonal vectors have a similarity of 0,
      and two opposite vectors have a similarity of -1.
       -(x1 * x2).sum(dim=1) is consistency loss, i.e. the negative cosine similarity.
  Reference:
      https://en.wikipedia.org/wiki/Cosine_similarity
  """
  x1 = F.normalize(x1, p=2., dim=-1, eps=1e-5)
  x2 = F.normalize(x2, p=2., dim=-1, eps=1e-5)
  return -(x1 * x2).sum(dim=1)

def cross_entropy_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
  return -(torch.log_softmax(prediction, dim=1) * target).sum(1)