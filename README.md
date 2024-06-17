# Pytorch implementation of Dirichlete Process Mixture Model (DPMM)

We integrate the Variational Inference framework with the autograd functionality of pytorch. To this end, we have extended pytorch autograd by defining a Function for the computation of DPMM:
- in the **forward pass**, we compute the cluster assignment and the elbo given the data and the model parameters;
- in the **bakcward pass**, we compute the natural gradient of the model parameters.

This allow to train DPMM without changing the training loop (which becomes the same of a neural model):
```
from torch_dpmm.models import GaussianDPMM
import torch.optim as optim

th_DPMM = GaussianDPMM(K=K, D=D, alphaDP=10, tau0=0, c0=1, n0=3*D, B0=1, is_diagonal=False)
optimiser = optim.SGD(params=m.parameters(), lr=0.1)

for i in range(num_epochs):
        optimiser.zero_grad()
        pi, elbo_loss = th_DPMM(x)
        elbo_loss.backward()
        optimiser.step()
```


We provide a complete example [here](https://github.com/danielecastellana22/torch_dpmm/blob/main/first_example.ipynb).
We recommend to download the notebook to view the animations.
