import os
import torch

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz
    
    A PyTorch implementation of Muon, designed for internal layers (e.g., linear/conv layers)
    where inputs are >= 2D. Note: Since we are using an Autoencoder (Linear layers),
    this fits well.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, backend='newton_schulz', backend_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, backend=backend, backend_steps=backend_steps)
        # Muon expects only >= 2D params.
        # We apply this to the weight matrices.
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim < 2:
                    # Fallback to SGD with momentum for 1D params (biases)
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        buf = state['momentum_buffer'] = torch.clone(g).detach()
                    else:
                        buf = state['momentum_buffer']
                        buf.mul_(momentum).add_(g)
                    
                    if group['nesterov']:
                        g = g.add(buf, alpha=momentum)
                    else:
                        g = buf
                    p.add_(g, alpha=-lr)
                    continue

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    buf = state['momentum_buffer'] = torch.clone(g).detach()
                else:
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)

                if group['nesterov']:
                    update = g.add(buf, alpha=momentum)
                else:
                    update = buf

                # Perform Orthogonalization (Newton-Schulz iteration)
                update = self._newton_schulz(update, steps=group['backend_steps'])

                # Scale the update
                update.mul_(lr * max(1, update.size(0) / update.size(1)) ** 0.5)

                p.add_(update, alpha=-1)

        return loss

    def _newton_schulz(self, G, steps=5):
        """
        Newton-Schulz iteration to refine the orthogonalization.
        """
        curr_device = G.device
        
        # Scale
        a, b, c = (3.4445, -4.7750,  2.0315)
        X = G.bfloat16() if G.dtype != torch.bfloat16 else G
        X /= (X.norm() + 1e-7)

        for _ in range(steps):
            A = X @ X.T
            B = A @ X
            X = a * X + b * B + c * A @ B

        return X.to(curr_device).type_as(G)
