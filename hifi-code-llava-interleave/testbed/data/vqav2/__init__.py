from typing import List, Union
from testbed.data.common import register_dataset_retriever, register_postprocess


@register_dataset_retriever(__name__.split(".")[-1])
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


# more post process will be done in evaluate procedure
@register_postprocess(__name__.split(".")[-1])
def postprocess(text: Union[str, List[str]]) -> Union[str, List[str]]:
    """
    Post-process generated text by removing special tokens like <end_of_outterance>
    """
    def clean_text(txt: str) -> str:
        # Remove <end_of_outterance> and its variations
        txt = txt.replace('<end_of_outterance>', '')
        txt = txt.replace('<end_of_outterance', '')
        # Also handle the incomplete tag without closing >
        if '<end_of_outter' in txt:
            txt = txt.split('<end_of_outter')[0]
        return txt.strip()
    
    if isinstance(text, str):
        return clean_text(text)
    elif isinstance(text, list):
        return [clean_text(t) for t in text]
    else:
        return text
