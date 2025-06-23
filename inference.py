import torch
from cascadia import Cascadia
from cascadia.data.flood import Flood

# Optionally set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the model (weights are mocked here; replace with real ones if available)
model = Cascadia(
    weights_backbone="none",  # Replace if you have local weights
    weights_design="none",  # Replace if you have local weights
    device=device,
    verbose=True
)

# Generate 1 sample flood map (default size 128x128)
sampled_flood = model.sample(
    samples=1,
    grid_size=[128, 128],  # spatial size (height, width)
    steps=250,             # diffusion steps
    full_output=False      # change to True for full denoising trajectory
)

# Visualize the output flood depth map
if isinstance(sampled_flood, Flood):
    sampled_flood.visualize()
else:
    sampled_flood[0].visualize()