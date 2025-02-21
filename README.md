# FiffDepth
code for "FiffDepth: Feed-forward Transformation of Diffusion-Based Generators for Detailed Depth Estimation"

## Inference
python run_direct.py \
    --checkpoint './checkpoints/fiffdepth' \
    --input_rgb_dir input/images \
    --output_dir output/depth
