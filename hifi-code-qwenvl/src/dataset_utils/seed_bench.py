import evaluate

import os
from datasets import load_dataset
from tqdm import tqdm

import src.paths as paths
from dataset_utils.interface import DatasetBase
from testbed.data import postprocess_generation
from utils import get_expand_runname


class Dataset(DatasetBase):
    support_datasets = ["seed_bench"]

    def __init__(self, data_cfg):
        super().__init__(data_cfg)
        ds_init_args = dict(
            path=os.path.join(paths.testbed_dir, "data", "seed_bench"),
            data_dir=paths.seed_dir,
        )

        assert (
            data_cfg.num_query_samples
        ), f"num_query_samples must be specified and greater than 0, but got {data_cfg.num_query_samples}"

        dataset = load_dataset(
            **ds_init_args, trust_remote_code=True, split="test"
        ).train_test_split(
            train_size=data_cfg.num_query_samples, seed=data_cfg.seed, shuffle=False
        )
        self._support_set = dataset["train"]
        self._query_set = dataset["test"]

    @property
    def num_role_in_round(self):
        # [
        #   { "role" : "image",
        #     "content" :  ... },
        #   { "role" : "question",
        #     "content" : ... },
        #   { "role" : "choices",
        #     "content" : ... },
        #   { "role" : "answer" }
        # ]
        return 4
    
    @staticmethod
    def metric_key():
        return "exact_match"

    def extract_answer(self, item):
        return item["answer"]

    @property
    def instruction(self):
        return None

    def eval(
        self,
        eval_cfg,
        model,
    ):
        result = []
        metric = evaluate.load("exact_match")
        eval_dl = self.validation_dataloader(eval_cfg.batch_size)
        iterations = eval_cfg.iterations or len(eval_dl)
        for _, batch in zip(
            range(iterations),
            tqdm(
                eval_dl,
                total=iterations,
                desc=f"Evaluating {model.model_name} with {get_expand_runname(eval_cfg)} ...",
            ),
        ):
            predictions = self.get_prediction(
                model,
                batch,
                max_skip_oom=eval_cfg.max_skip_oom,
                **eval_cfg.generation_args,
            )
            if predictions is None:
                continue

            for pred, context in zip(predictions, batch):
                last_qa = context[-1]
                prediction = postprocess_generation(
                    self.name, pred, stop_words=["\n", "."]
                )
                if prediction.upper() not in ["A", "B", "C", "D"]:
                    prediction = random.choice(["A", "B", "C", "D"])
                gt_answer = last_qa["answer"]
                metric.add(
                    prediction=prediction,
                    reference=gt_answer,
                )
                result.append(
                    {
                        "question": last_qa["question"],
                        "question_id": last_qa["question_id"],
                        "raw_output": pred,
                        "question": last_qa["question"],
                        "choice_a": last_qa["choice_a"],
                        "choice_b": last_qa["choice_b"],
                        "choice_c": last_qa["choice_c"],
                        "choice_d": last_qa["choice_d"],
                        "prediction": prediction,
                        "answer": last_qa["answer"],
                    }
                )

        return result, metric.compute()
