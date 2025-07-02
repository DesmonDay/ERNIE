# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

nohup sh examples/post-training/vl_sft/script/train_gpu_single.sh \
--from_scratch 0 \
--vision_from_scratch 0 \
--vit_lr_ratio 0.9 \
--freeze_config "freeze_vision" \
--multimodal true \
--vision_model_name_or_path "/root/paddlejob/workspace/env_run/output/wenxinhao/weights/sft/10.95.117.207:8000/dfn_stage3tie_46k_FC" \
--model_name_or_path "/root/paddlejob/workspace/env_run/output/huangkui/weight/eb45lm_afterStage3128k_v11_think_32k" \
--tokenizer_name "data_processor/tokenizer_models/ernie4_tk_m100k_103367_sft_think" \
--output_dir "./output/lite_10step_erniekit" \
--data_load_process_num 1 \
--max_seq_length 8192 \
--pad_to_max_seqlen 8192 \
--base_seq_length 8192 \
--variable_resolution 1 \
--global_shuffle_num_examples 1000000 \
--sequence_parallel 1 \
--moe_use_aux_free_update_coef 0.0 \
--data_filelist /root/paddlejob/workspace/env_run/output/wenxinhao/SFT_45V/data_config/mm_sft/data_config_debug_json_mul.yaml \
--visual_ld 0.9 \
--modality_ratio [1,1] \
--moe_gate_lr_ratio 0.01 \
--dataloader_num_workers 1 \
--dataset_name "KnowledgeBasedSFTReader" \
--text_sft_task_config "/root/paddlejob/workspace/env_run/output/wenxinhao/SFT_45V/data_config/text_sft/data_config_0120_v1_test.json" \
--add_sys_token true \
--number_of_samples_each_epoch 10000000 \
--drop_history_with_k true \
--prefetch_factor 4 \
--one_sample_in_one_seq true \
--use_multi_image_crop true \
--serialize_output false \
--chat_template deepseek \
--render_timestamp true \
--pp_need_data true \
--adam_beta2 0.95 \
--bf16 true \
--do_train true \
--fp16_opt_level O2 \
--global_batch_size 8 \
--learning_rate 1e-05 \
--logging_steps 1 \
--lr_scheduler_type cosine \
--max_steps 1 \
--min_lr 1e-06 \
--same_data true \
--load_sharded_model true \
--save_sharded_model true \
--scale_loss 4096 \
--warmup_steps 1 \
--weight_decay 0.1 \
--overwrite_output_dir 1 \
--per_device_eval_batch_size 1 \
--per_device_train_batch_size 1 \
--pp_need_data_degree 2 \
--pipeline_parallel_degree 2 \
--tensor_parallel_degree 4 \
--amp_master_grad 1 \
--pipeline_parallel_config "enable_offload_queue enable_delay_scale_loss enable_overlap_p2p_comm best_unbalanced_scheduler" \
--sharding_parallel_config "split_param enable_fuse_optimizer_states" \
--tensor_parallel_config "sync_param sync_grad sync_moment" \
--pre_alloc_memory 30 \
--sharding_comm_buffer_size_MB 512 \
--use_moe true \
--moe_group mp \
--gc_interval 100000 \
--skip_profile_timer 0 \
--save_sharding_stage1_model_include_freeze_params true \
--disable_pipeline_warmup true \
> lite_erniekits_1step.log 2>err &