from datasets import load_dataset
import evaluate
from tqdm import tqdm

import src.paths as paths
from dataset_utils.interface import DatasetBase
from utils import get_expand_runname
from testbed.data import (
    postprocess_generation,
    register_dataset_retriever,
    register_postprocess,
)


@register_dataset_retriever("mme")
def retriever(item, is_last: bool):
    return (
        [
            {"role": "image", "content": [{"type": "image"}]},
            {
                "role": "question",
                "content": [{"type": "text", "text": item["question"]}],
            },
            (
                {"role": "answer"}
                if is_last
                else {
                    "role": "answer",
                    "content": [{"type": "text", "text": item["answer"]}],
                }
            ),
        ],
        item["image"],
    )


@register_postprocess("mme")
def postprocess(text):
    if text.lower() == "yes":
        return 1
    elif text.lower() == "no":
        return 0
    return -1


class Dataset(DatasetBase):
    support_datasets = ["mme"]

    def __init__(self, data_cfg):
        super().__init__(data_cfg)
        ds_init_args = dict(path="parquet", data_dir=paths.mme_dir)

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
        # the last 3 items:
        # [
        #   { "role" : "image",
        #     "content" :  ... },
        #   { "role" : "question",
        #     "content" : ... },
        #   { "role" : "answer" }
        # ]
        return 3
    
    @staticmethod
    def metric_key():
        return "accuracy"

    def extract_answer(item):
        return item["answer"]

    @property
    def instruction(self):
        return 'Provide an answer in "Yes" or "No" to the question. Use the image to answer.'

    def eval(
        self,
        eval_cfg,
        model,
    ):
        result = []
        metric = evaluate.load("accuracy")
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
                gt_answer = last_qa["answer"]
                prediction = postprocess_generation(self.name, pred, stop_words=["\n"])
                metric.add(prediction=prediction, reference=gt_answer.lower() == "yes")
                result.append(
                    {
                        "prediction": prediction,
                        "answer": gt_answer,
                        "question": last_qa["question"],
                        "raw_output": pred,
                        "question_id": last_qa["question_id"],
                    }
                )

        return result, metric.compute()
