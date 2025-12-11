#!/bin/bash
set -e


python examples/test_weather_svae.py \
    --data-path gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr \
    --model-path outputs/svae_vmf/best_model.pth \
    --variable 2m_temperature \
    --time-slice 2020-01-01:2020-12-31 \
    --output-dir outputs/svae_vmf_test