#!/bin/bash

# python predict.py \
#   --model_a_path s1_glint360k_r50_512d_gmdb__v1.1.0_bs64_size112_channels3_last_model.pth \
#   --model_b_path s2_glint360k_r100_512d_gmdb__v1.1.0_bs128_size112_channels3_last_model.pth \
#   --model_c_path glint360k_r100.onnx \
#   --save_as_pickle \
#   --data ./demo_images/cdls_demo_aligned.jpg \
#   --save_dir ./data/demo_test/ \
#   --output_name test_encodings_v1.1.0.pkl \
#   --weight_dir ./saved_models \

python evaluate.py \
  --metadata_dir ./data/GestaltMatcherDB/v1.1.0/gmdb_metadata \
  --gallery_input ./data/gallery_encodings/GMDB_gallery_encodings_v1.1.0.pkl \
  --case_input ./data/demo_test/test_encodings_v1.1.0.pkl \
  --output_dir demo_output \
  --output_file demo_results.json \
  --top_n all