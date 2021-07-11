# #!/usr/bin/env bash
# Author: Chunyu Xue

# MIG 4g.20gb Device 0: (UUID: MIG-GPU-9de3d0e8-33f5-10dc-0c79-2c88a7ab0a23/2/0)
python main.py --min_batch_size 1 --max_batch_size 128 --learning_rate 3e-4 --gpu_device MIG-GPU-9de3d0e8-33f5-10dc-0c79-2c88a7ab0a23/2/0

# MIG 2g.10gb Device 1: (UUID: MIG-GPU-9de3d0e8-33f5-10dc-0c79-2c88a7ab0a23/3/0)
python main.py --min_batch_size 1 --max_batch_size 128 --learning_rate 3e-4 --gpu_device MIG-GPU-9de3d0e8-33f5-10dc-0c79-2c88a7ab0a23/3/0

# MIG 1g.5gb Device 2: (UUID: MIG-GPU-9de3d0e8-33f5-10dc-0c79-2c88a7ab0a23/9/0)
python main.py --min_batch_size 1 --max_batch_size 128 --learning_rate 3e-4 --gpu_device MIG-GPU-9de3d0e8-33f5-10dc-0c79-2c88a7ab0a23/9/0
