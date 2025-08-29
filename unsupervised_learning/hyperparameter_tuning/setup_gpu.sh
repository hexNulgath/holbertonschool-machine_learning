#!/bin/bash
# Setup script for GPU support in TensorFlow

# Set CUDA library paths
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/miniconda3/envs/tf_env/lib/python3.10/site-packages/nvidia/cudnn/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/miniconda3/envs/tf_env/lib/python3.10/site-packages/nvidia/cublas/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/miniconda3/envs/tf_env/lib/python3.10/site-packages/nvidia/cufft/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/miniconda3/envs/tf_env/lib/python3.10/site-packages/nvidia/curand/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/miniconda3/envs/tf_env/lib/python3.10/site-packages/nvidia/cusolver/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/miniconda3/envs/tf_env/lib/python3.10/site-packages/nvidia/cusparse/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/miniconda3/envs/tf_env/lib/python3.10/site-packages/nvidia/cuda_runtime/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/miniconda3/envs/tf_env/lib/python3.10/site-packages/nvidia/cuda_cupti/lib

echo "GPU environment setup complete!"
echo "Current LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
