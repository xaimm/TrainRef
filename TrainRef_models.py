import torch
import torch.nn as nn

class LastLayer(nn.Module):
    def __init__(self, in_features=512, out_features=10, external_last_layer=None):
        super(LastLayer, self).__init__()

        # If an external model is provided, configure in_features and out_features based on it
        if external_last_layer is not None:
            if isinstance(external_last_layer, nn.Linear):
                in_features = external_last_layer.in_features
                out_features = external_last_layer.out_features
            else:
                raise ValueError("External model must be an instance of nn.Linear")
        
        # Initialize the fully connected layer
        self.fc = nn.Linear(in_features, out_features)
        
        # Optionally, load weights and biases from the external model
        if external_last_layer is not None:
            self.load_from_external_model(external_last_layer)

        # Register hooks
        self.fc.register_forward_hook(self.forward_hook)
        self.fc.register_full_backward_hook(self.backward_hook)

    def forward(self, x):
        out = self.fc(x)
        return out
    
    def forward_hook(self, module, input, output):
        module.activations = input[0]

    def backward_hook(self, module, grad_input, grad_output):
        if len(grad_output[0].shape) > 2:
            averaged_grad_output = torch.mean(grad_output[0], dim=(2, 3))
            averaged_activations = torch.mean(module.activations, dim=(2, 3))
        else:
            averaged_grad_output = grad_output[0]
            averaged_activations = module.activations
        module.grad_sample = torch.einsum('n...i,n...j->nij', averaged_grad_output, averaged_activations)
        
    def get_persample_grad(self):
        fc_grad = self.fc.grad_sample
        b_s = fc_grad.shape[0]
        fc_grad = fc_grad.view(b_s, -1).detach().clone()
        return fc_grad
    
        
    def load_from_external_model(self, external_last_layer):
        # This method now expects that external_last_layer is a proper instance of nn.Linear
        self.fc.weight.data = external_last_layer.weight.data.clone()
        self.fc.bias.data = external_last_layer.bias.data.clone()