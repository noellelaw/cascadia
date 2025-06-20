# cascadia

![Overview](https://raw.githubusercontent.com/noellelaw/cascadia/main/assets/THRUST3_Overview.png)
This framework (will eventually) support both unconditional and conditional flood generation using 
diffusion-based backbones and graph-based refinement networks. It enables physically 
plausible, programmable flood scenario design under varying sea level rise (SLR), 
storm intensities, and topographic constraints.

Applications include:
    - Generating high-resolution synthetic flood extents across diverse geographic regions
    - Conditioning flood generation on climate drivers (e.g., SLR, wind fields, rainfall)
    - Propagating cascading infrastructure failures via graph neural networks
    - Supporting scenario planning for resilient infrastructure and emergency response

DDPM Head
![DDPM Head](https://raw.githubusercontent.com/noellelaw/cascadia/main/assets/DDPM_Scenario_Overview.png)

GNN-Design Network for Flood Propogation in High- and Low-Resource Areas
![GNN A]((https://raw.githubusercontent.com/noellelaw/cascadia/main/assets/DDPM_GNN_LowResource_Scenario_Overview.png)

GNN-Design Network for Cascading Failures in Interconnected Infrasructure
![GNN B](https://raw.githubusercontent.com/noellelaw/cascadia/main/assets/DDPM_GNN_IIC_Scenario_Overview.png)


Code adapted from the protein synthesis space (thank you Chroma: [Paper](https://www.nature.com/articles/s41586-023-06728-8), [GitHub]())
