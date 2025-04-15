
from transformers import AutoTokenizer
from typing import List, Union, Dict, Optional

class MiniMindTokenizer:
    def __init__(self, pretrained_tokenizer_name: str = "minimind_tokenizer"):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_name)
        
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        padding: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
    ) -> Union[List[int], Dict[str, List[int]]]:

        encoding = self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            padding=padding,
            max_length=max_length,
            return_tensors=return_tensors,
        )
        
        # 如果return_tensors=None，返回List[int]，否则返回字典
        if return_tensors is None:
            return encoding["input_ids"] if isinstance(text, str) else encoding
        return encoding

    def decode(
        self,
        token_ids: Union[int, List[int], "torch.Tensor", "tf.Tensor"],
        skip_special_tokens: bool = True,
    ) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    


if __name__=='__main__':
    pass
    # 初始化tokenizer
    tokenizer = MiniMindTokenizer("minimind_tokenizer")
    text = "Hello, world!"
    encoded = tokenizer.encode(text)
    print("Encoded:", encoded)  # 例如: [101, 7592, 1010, 2088, 102]

    # 解码示例
    decoded = tokenizer.decode(encoded)
    print("Decoded:", decoded)  # "Hello, world!"