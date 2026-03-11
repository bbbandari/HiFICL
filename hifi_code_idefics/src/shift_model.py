import enum
from functools import reduce
import json
import os
import sys

from omegaconf import OmegaConf
import evaluate

sys.path.insert(0, "..")
import paths
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from deepspeed.ops.adam import DeepSpeedCPUAdam
from transformers import get_cosine_schedule_with_warmup

from utils import save_pretrained, get_expand_runname
from testbed.data import postprocess_generation


class Strategy(enum.IntFlag):
    LAYER_WISE_MSE = 2
    LAYER_WISE_COS_SIM = 64  # equivalent to normalized L2 distance
    LOGITS_KL_DIV = 4
    LM_LOSS = 8
    SYNERGISTIC_LOSS = 128  # 协同损失

    def has_layer_wise(self):
        try:
            self.layer_wise_strategy()
            return True
        except ValueError:
            return False

    def validate(self):
        layer_wise_loss = [
            Strategy.LAYER_WISE_MSE,
            Strategy.LAYER_WISE_COS_SIM,
        ]

        if bin(self & reduce(lambda x, y: x | y, layer_wise_loss)).count("1") > 1:
            raise ValueError(
                f"{[e.name for e in layer_wise_loss]} are mutually exclusive."
            )

    def layer_wise_strategy(self):
        if Strategy.LAYER_WISE_MSE in self:
            return "mse_loss"
        elif Strategy.LAYER_WISE_COS_SIM in self:
            return "cos_sim"
        else:
            raise ValueError("None of layer wise loss strategy is enabled")


class ShiftModel(pl.LightningModule):
    def __init__(
        self,
        cfg,
        shift_encoder,
        strategy: Strategy,
        save_checkpoint_when=None,  # should be a method f(epoch), save the last ckpt by default
    ) -> None:
        super().__init__()
        self.lmm = shift_encoder.lmm
        self.cfg = cfg
        self.shift_encoder = shift_encoder
        strategy.validate()
        self.strategy = strategy
        self.save_checkpoint_when = (
            save_checkpoint_when
            if save_checkpoint_when is not None
            else lambda epoch: epoch == self.trainer.max_epochs - 1
        )
        self.save_dir = os.path.join(paths.result_dir, "ckpt", get_expand_runname(cfg))

    def generate_label_mask(self, inputs, num_separator, keep_bos=False):
        """
        Generates label mask which masks tokens before num_separator pad_tokens from given inputs.
        """
        input_ids = inputs["input_ids"]
        batch_size, seq_len = input_ids.shape
        pad_mask = input_ids == self.lmm.processor.tokenizer.pad_token_id
        non_pad_mask = ~pad_mask
        label_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        if self.lmm.processor.tokenizer.padding_side == "left":
            bos_position = non_pad_mask.long().argmax(dim=1)

        for i in range(batch_size):
            seq_pad_positions = pad_mask[i].nonzero(as_tuple=False).squeeze(-1)

            if self.lmm.processor.tokenizer.padding_side == "left":
                seq_pad_positions = seq_pad_positions[
                    seq_pad_positions > bos_position[i]
                ]

            num_pads = len(seq_pad_positions)
            if num_pads < num_separator:
                raise ValueError(
                    f"Sequence {i} has fewer pad tokens ({num_pads}) than num_separator ({num_separator})"
                )

            sep_position = seq_pad_positions[num_separator - 1].item()
            label_mask[i, sep_position + 1 :] = True

        label_mask = label_mask & non_pad_mask
        if keep_bos:
            label_mask[torch.arange(batch_size, device=self.device), bos_position] = (
                True
            )

        return label_mask

    def remove_hooks(self, hooks):
        # remove all hooks
        for name, handles in hooks.items():
            if isinstance(handles, list):
                for handle in handles:
                    handle.remove()
            else:
                handles.remove()

    def get_hidden_states(self, query_label_mask):
        """
        Apply query_label_mask to extract query parts from hidden states (shape: num_layer * [batch_size, seq_len, d_model]),
        and convert to batch_size * [num_layer, query_part_len, d_model].
        """
        hidden_states_dict = {}

        for name, attr in vars(self.shift_encoder).items():
            if "hidden_states" in name:
                # [num_layer, batch_size, seq_len, d_model] -> [batch_size, num_layer, seq_len, d_model]
                hidden_states = torch.stack(attr).transpose(0, 1)
                batch_size, num_layer, seq_len, d_model = hidden_states.shape
                hidden_states_dict[name] = [
                    hs.masked_select(mask[None, :, None]).view(num_layer, -1, d_model)
                    for hs, mask in zip(hidden_states, query_label_mask)
                ]

        if not hidden_states_dict:
            raise RuntimeError(
                "Layer wise loss requires to record hidden states, but no any *_hidden_states in shift encoder."
            )

        return hidden_states_dict

    def calculate_layer_wise_loss(self, shift_hidden_states, prefix_hidden_states):
        if Strategy.LAYER_WISE_MSE in self.strategy:
            loss_fn = lambda input, target: F.mse_loss(
                input,
                target,
                reduction="mean",
            )
        elif Strategy.LAYER_WISE_COS_SIM in self.strategy:
            loss_fn = lambda input, target: 1 - torch.mean(
                F.cosine_similarity(
                    input,
                    target,
                    dim=-1,
                ),
                dim=1,
            )

        layer_loss = dict()
        for (shift_hs_varname, shift_hs_list), (_, prefix_hs_list) in zip(
            shift_hidden_states.items(), prefix_hidden_states.items()
        ):
            # hs_list: batch_size * [num_layer, query_part_len, d_model]
            layer_loss[
                shift_hs_varname.replace(
                    "hidden_states", self.strategy.layer_wise_strategy()
                )
            ] = torch.mean(
                torch.stack(
                    [
                        loss_fn(shift_hs, prefix_hs)
                        for shift_hs, prefix_hs in zip(shift_hs_list, prefix_hs_list)
                    ]
                )
            )
        return layer_loss

    def calculate_logits_kl_loss(
        self, shift_logits, prefix_logits, query_label_inputs, prefix_label_mask
    ):
        # extract answer [EOS]
        logits_kl_loss = F.kl_div(
            shift_logits[query_label_inputs].log_softmax(dim=-1),
            prefix_logits[prefix_label_mask].softmax(dim=-1),
            reduction="batchmean",
            log_target=False,
        )
        return {"logits_kl_loss": logits_kl_loss}

    def calculate_synergistic_loss(self, gamma):
        """
        Calculates L_syn based on the normalized outputs of MHA and MLP branches
        from the shift_encoder.
        """
        if not hasattr(self.shift_encoder, 'mha_outputs_for_syn') or not hasattr(self.shift_encoder, 'mlp_outputs_for_syn'):
            return None

        mha_outputs = self.shift_encoder.mha_outputs_for_syn
        mlp_outputs = self.shift_encoder.mlp_outputs_for_syn
        
        if not mha_outputs or not mlp_outputs or len(mha_outputs) != len(mlp_outputs):
            # 清理以防万一
            if hasattr(self.shift_encoder, 'mha_outputs_for_syn'): 
                self.shift_encoder.mha_outputs_for_syn = []
            if hasattr(self.shift_encoder, 'mlp_outputs_for_syn'): 
                self.shift_encoder.mlp_outputs_for_syn = []
            return None

        total_syn_loss = 0.0
        num_layers = len(mha_outputs)

        for l in range(num_layers):
            # 注意：MHA 输出形状是 [bsz, nh, t, nd]，MLP 是 [bsz, t, hidden_dim]
            # 需要将 MHA reshape 以便计算
            mha_out = mha_outputs[l]
            bsz, nh, t, nd = mha_out.shape
            mha_out_reshaped = mha_out.transpose(1, 2).reshape(bsz, t, -1)
            
            mlp_out = mlp_outputs[l]

            # 归一化 (只对 query 部分)
            # 假设 mha_out 和 mlp_out 已经是被 mask 过的 query 部分
            z_mha = F.normalize(mha_out_reshaped, p=2, dim=-1)
            z_mlp = F.normalize(mlp_out, p=2, dim=-1)

            # 计算 cross-view correlation matrix M^l (bsz, hidden_dim, hidden_dim)
            M_l = torch.bmm(z_mha.transpose(1, 2), z_mlp)
            
            # 计算损失
            identity = torch.eye(M_l.shape[-1], device=self.device)
            loss_diag = F.mse_loss(M_l.diagonal(dim1=-2, dim2=-1), identity.diagonal().expand_as(M_l.diagonal(dim1=-2, dim2=-1)))
            
            # Off-diagonal elements loss
            loss_off_diag = (M_l - torch.diag_embed(M_l.diagonal(dim1=-2, dim2=-1))).pow(2).mean()

            total_syn_loss += loss_diag + gamma * loss_off_diag
        
        # 清理记录，为下一个 batch 做准备
        self.shift_encoder.mha_outputs_for_syn = []
        self.shift_encoder.mlp_outputs_for_syn = []
        
        return total_syn_loss / num_layers

    def evaluate_teacher_correctness(self, prefix_texts, query_texts, answers, dataset_name="vqav2"):
        """
        评估教师模型对当前batch的回答是否正确。
        在训练过程中，prefix_texts已经包含了教师模型的完整输出（ICE + query + teacher_answer），
        我们只需要从中提取教师模型的回答部分并与ground truth进行比较。
        返回一个布尔列表，表示每个样本的教师模型是否回答正确。
        """
        if not self.cfg.get("teacher_eval_enabled", False):
            # 如果未启用教师模型评估，默认所有样本都正确
            return [True] * len(query_texts)
        
        correctness_list = []
        
        for i, (prefix_text, query_text, answer) in enumerate(zip(prefix_texts, query_texts, answers)):
            # 从prefix_text中提取教师模型的回答
            teacher_answer = self._extract_teacher_answer(prefix_text, query_text)
            
            # 后处理教师模型的回答
            processed_teacher_answer = self._postprocess_answer(teacher_answer, dataset_name)
            
            # 判断教师模型回答是否正确
            is_correct = self._check_answer_correctness(
                processed_teacher_answer, answer, dataset_name
            )
            
            correctness_list.append(is_correct)
        
        return correctness_list
    
    def _extract_teacher_answer(self, prefix_text, query_text):
        """
        从prefix_text中提取教师模型的回答部分。
        prefix_text格式: ICE + query + teacher_answer
        """
        # 找到query_text在prefix_text中的位置
        query_pos = prefix_text.find(query_text)
        if query_pos != -1:
            # 找到query之后的部分
            after_query = prefix_text[query_pos + len(query_text):]
            # 清理可能的pad token和特殊字符
            teacher_answer = after_query.strip()
            
            # 移除可能的pad token
            pad_token = self.lmm.processor.tokenizer.pad_token
            if pad_token and teacher_answer.startswith(pad_token):
                teacher_answer = teacher_answer[len(pad_token):].strip()
            
            # 移除可能的eos token
            eos_token = self.lmm.processor.tokenizer.eos_token
            if eos_token and teacher_answer.endswith(eos_token):
                teacher_answer = teacher_answer[:-len(eos_token)].strip()
                
        else:
            # 如果找不到query，使用整个prefix_text作为教师回答
            teacher_answer = prefix_text.strip()
        
        return teacher_answer
    
    def _postprocess_answer(self, answer, dataset_name):
        """
        后处理答案，移除不需要的字符。
        """
        # 移除常见的无用字符
        remove_list = ["\n", "Question", "Answer", "Image", "Short"]
        processed_answer = answer
        for remove_str in remove_list:
            processed_answer = processed_answer.replace(remove_str, "")
        
        return processed_answer.strip()
    
    def _check_answer_correctness(self, prediction, ground_truth, dataset_name):
        """
        检查预测答案是否正确。
        """
        if dataset_name == "vqav2":
            # VQA使用多个答案，需要特殊处理
            if isinstance(ground_truth, str):
                gt_answers = [ground_truth]
            else:
                gt_answers = ground_truth if isinstance(ground_truth, list) else [str(ground_truth)]
            
            # VQA的准确率计算：预测答案是否在ground truth列表中
            prediction_lower = prediction.lower().strip()
            gt_answers_lower = [ans.lower().strip() for ans in gt_answers]
            return prediction_lower in gt_answers_lower
        else:
            # 其他数据集使用exact match
            prediction_lower = prediction.lower().strip()
            gt_lower = str(ground_truth).lower().strip()
            return prediction_lower == gt_lower

    def forward(self, prefix_texts, query_texts, answers, images):
        pad_token, pad_token_id, bos_token_id, eos_token = (
            self.lmm.processor.tokenizer.pad_token,
            self.lmm.processor.tokenizer.pad_token_id,
            self.lmm.processor.tokenizer.bos_token_id,
            self.lmm.processor.tokenizer.eos_token,
        )
        loss_dict = {"loss": 0.0}
        hooks = self.shift_encoder.register_record_hooks()

        # prepare inputs
        query_answer = [
            query + pad_token + answer + eos_token
            for query, answer in zip(query_texts, answers)
        ]
        query_images = [img[-self.cfg.data.num_image_in_query :] for img in images]
        query_inputs = self.lmm.process_input(query_images, query_answer).to(
            device=self.device, dtype=self.dtype
        )
        query_inputs["attention_mask"] = query_inputs["input_ids"] != pad_token_id
        if self.strategy != Strategy.LM_LOSS:
            # if strategy only has lm_loss, full context forward is no need
            full_text = [
                ice + pad_token + query + pad_token + answer + eos_token
                for ice, query, answer in zip(prefix_texts, query_texts, answers)
            ]
            inputs = self.lmm.process_input(images, full_text).to(
                device=self.device, dtype=self.dtype
            )
            inputs["attention_mask"] = inputs["input_ids"] != pad_token_id

            # step 1. [SOS](implicitly added) ICE [PAD] query [PAD] answer [EOS] forward process
            with torch.no_grad(), self.lmm.model.disable_adapter():
                prefix_logits = self.lmm.model(**inputs)["logits"]

            # extract query + [PAD] + answer + [EOS]
            prefix_hidden_states = (
                self.get_hidden_states(self.generate_label_mask(inputs, 1))
                if self.strategy.has_layer_wise()
                else None
            )
            prefix_label_mask = self.generate_label_mask(inputs, 2)

        # step 2. [SOS](implicitly added) + query + [PAD] + answer [EOS] forward process
        hooks.update(self.shift_encoder.register_shift_hooks())
        query_outputs = self.lmm.model(
            **query_inputs,
            labels=(
                query_inputs["input_ids"] if Strategy.LM_LOSS in self.strategy else None
            ),
        )
        shift_logits = query_outputs["logits"]
        if Strategy.LM_LOSS in self.strategy:
            loss_dict["ce_loss"] = query_outputs["loss"]
            ce_loss_weight = (
                1.0 if self.strategy == Strategy.LM_LOSS else self.cfg.ce_loss_weight
            )
            loss_dict["loss"] += ce_loss_weight * query_outputs["loss"]

        # extract query + answer + [EOS]
        shift_hidden_states = (
            self.get_hidden_states(
                query_inputs["attention_mask"]
                & (query_inputs["input_ids"] != bos_token_id)
            )
            if self.strategy.has_layer_wise()
            else None
        )

        self.remove_hooks(hooks)

        # step 3. evaluate teacher correctness and calculate layer-wise loss conditionally
        if self.strategy.has_layer_wise():
            # 评估教师模型的正确性
            teacher_correctness = self.evaluate_teacher_correctness(
                prefix_texts, query_texts, answers, 
                dataset_name=getattr(self.cfg.data, 'name', 'vqav2')
            )
            
            # 记录教师模型正确性统计
            correct_count = sum(teacher_correctness)
            total_count = len(teacher_correctness)
            loss_dict["teacher_correctness_rate"] = correct_count / total_count if total_count > 0 else 0.0
            
            # 只有当教师模型回答正确时才计算中间状态对齐损失
            if correct_count > 0:
                # 过滤出教师模型回答正确的样本
                correct_indices = [i for i, is_correct in enumerate(teacher_correctness) if is_correct]
                
                # 过滤hidden states
                filtered_shift_hidden_states = {}
                filtered_prefix_hidden_states = {}
                
                for (shift_hs_varname, shift_hs_list), (prefix_hs_varname, prefix_hs_list) in zip(
                    shift_hidden_states.items(), prefix_hidden_states.items()
                ):
                    filtered_shift_hidden_states[shift_hs_varname] = [
                        shift_hs_list[i] for i in correct_indices
                    ]
                    filtered_prefix_hidden_states[prefix_hs_varname] = [
                        prefix_hs_list[i] for i in correct_indices
                    ]
                
                layer_loss = self.calculate_layer_wise_loss(
                    filtered_shift_hidden_states, filtered_prefix_hidden_states
                )
                loss_dict.update(layer_loss)
                loss_dict["loss"] += self.cfg.align_loss_weight * sum(layer_loss.values())
                
                # 记录实际用于对齐的样本数量
                loss_dict["aligned_samples"] = correct_count
            else:
                # 如果教师模型全部回答错误，不计算对齐损失
                loss_dict["aligned_samples"] = 0
                loss_dict["layer_wise_loss"] = 0.0

        # step 4. calculate the last logits kl div (only for correct teacher predictions)
        if Strategy.LOGITS_KL_DIV in self.strategy:
            # 如果启用了教师模型评估，需要检查教师模型正确性
            if self.cfg.get("teacher_eval_enabled", False):
                if 'teacher_correctness' not in locals():
                    teacher_correctness = self.evaluate_teacher_correctness(
                        prefix_texts, query_texts, answers, 
                        dataset_name=getattr(self.cfg.data, 'name', 'vqav2')
                    )
                
                correct_count = sum(teacher_correctness)
                if correct_count > 0:
                    # 过滤logits和masks
                    correct_indices = [i for i, is_correct in enumerate(teacher_correctness) if is_correct]
                    
                    # 过滤shift_logits和prefix_logits
                    filtered_shift_logits = shift_logits[correct_indices]
                    filtered_prefix_logits = prefix_logits[correct_indices]
                    
                    # 过滤masks
                    query_mask = self.generate_label_mask(query_inputs, 1)
                    filtered_query_mask = query_mask[correct_indices]
                    filtered_prefix_mask = prefix_label_mask[correct_indices]
                    
                    logits_kl_loss = self.calculate_logits_kl_loss(
                        filtered_shift_logits,
                        filtered_prefix_logits,
                        filtered_query_mask,
                        filtered_prefix_mask,
                    )
                    loss_dict.update(logits_kl_loss)
                    loss_dict["loss"] += self.cfg.align_loss_weight * sum(
                        logits_kl_loss.values()
                    )
                else:
                    loss_dict["logits_kl_loss"] = 0.0
            else:
                # 如果未启用教师模型评估，按原逻辑计算
                logits_kl_loss = self.calculate_logits_kl_loss(
                    shift_logits,
                    prefix_logits,
                    self.generate_label_mask(query_inputs, 1),
                    prefix_label_mask,
                )
                loss_dict.update(logits_kl_loss)
                loss_dict["loss"] += self.cfg.align_loss_weight * sum(
                    logits_kl_loss.values()
                )

        # step 5. calculate synergistic loss
        if Strategy.SYNERGISTIC_LOSS in self.strategy:
            # 从配置文件中获取 gamma 超参数
            gamma = self.cfg.get("gamma", 1.0) # 提供一个默认值
            syn_loss = self.calculate_synergistic_loss(gamma)
            if syn_loss is not None:
                loss_dict["syn_loss"] = syn_loss
                # 从配置文件中获取协同损失的权重
                syn_loss_weight = self.cfg.get("syn_loss_weight", 0.1)
                loss_dict["loss"] += syn_loss_weight * syn_loss

        return loss_dict

    def training_step(self, batch, batch_idx):
        loss_dict = self.forward(**batch)
        self.log_dict(loss_dict, sync_dist=True, prog_bar=True)

        return loss_dict["loss"]

    def on_train_epoch_end(self):
        if self.save_checkpoint_when(self.current_epoch) and self.global_rank == 0:
            save_pretrained(
                os.path.join(self.save_dir, f"epoch-{self.current_epoch}"),
                self.lmm,
                self.shift_encoder,
            )

    def on_train_end(self):
        if self.global_rank == 0:
            with open(os.path.join(self.save_dir, "config.json"), "w") as f:
                json.dump(OmegaConf.to_container(self.cfg, resolve=True), f, indent=4)

    def configure_optimizers(self):
        def filter_decay_params(param_dict, **common_args):
            """filter parameters for optimizer, separate parameters by adding weight_decay or not"""
            non_decay_names = ["bias"]
            non_decay = [
                {
                    "params": [
                        p
                        for n, p in param_dict.items()
                        for name in non_decay_names
                        if name in n
                    ],
                    "weight_decay": 0.0,
                    **common_args,
                }
            ]

            decay = [
                {
                    "params": [
                        p
                        for n, p in param_dict.items()
                        for name in non_decay_names
                        if name not in n
                    ],
                    "weight_decay": self.cfg.weight_decay,
                    **common_args,
                }
            ]

            return [*non_decay, *decay]

        param_dict = {
            n: p for n, p in self.shift_encoder.named_parameters() if p.requires_grad
        }
        
        # 检查是否启用了分支学习率配置 (用于 NLICV/M2IV)
        if self.cfg.get("mha_lr") and self.cfg.get("ffn_lr"):
            # 分离 MHA 和 FFN 参数
            mha_params = {
                n: p for n, p in param_dict.items() 
                if any(keyword in n for keyword in ['attn', 'gate_generator', 'context_vector'])
            }
            ffn_params = {
                n: p for n, p in param_dict.items() 
                if any(keyword in n for keyword in ['ffn', 'mlp', 'scale_vector', 'shift_vector'])
            }
            other_params = {
                n: p for n, p in param_dict.items() 
                if n not in mha_params and n not in ffn_params
            }

            optim_groups = [
                *filter_decay_params(mha_params, lr=self.cfg.mha_lr),
                *filter_decay_params(ffn_params, lr=self.cfg.ffn_lr),
                *filter_decay_params(other_params, lr=self.cfg.lr),
            ]
        elif self.cfg.peft.get("scale_lr", None):
            # if scale_lr is provided, separate scale parameters and regular parameters
            # scale parameters will have a different learning rate, which typically is
            # used for LIVE.
            scale_params = {
                n: p for n, p in param_dict.items() if "log_Z1" in n or "scale" in n
            }
            regular_params = {
                n: p for n, p in param_dict.items() if n not in scale_params
            }

            optim_groups = [
                *filter_decay_params(regular_params, lr=self.cfg.lr),
                *filter_decay_params(scale_params, lr=self.cfg.peft.scale_lr),
            ]
        else:
            optim_groups = filter_decay_params(param_dict, lr=self.cfg.lr)

        assert any(
            group["params"] is not None for group in optim_groups if "params" in group
        ), "No parameter to optimize."

        if "deepspeed" in self.cfg.strategy:
            optimizer = DeepSpeedCPUAdam(
                optim_groups,
                weight_decay=self.cfg.weight_decay,
            )
        else:
            optimizer = optim.AdamW(
                optim_groups,
                weight_decay=self.cfg.weight_decay,
            )

        step_batches = self.trainer.estimated_stepping_batches
        warmup_steps = self.cfg.warmup_step
        if isinstance(warmup_steps, float):
            warm_steps = warmup_steps * step_batches
        elif isinstance(warmup_steps, int):
            warm_steps = warmup_steps
        else:
            raise ValueError(
                f"the warm_steps should be int or float, but got {type(warmup_steps)}"
            )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warm_steps, num_training_steps=step_batches
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
