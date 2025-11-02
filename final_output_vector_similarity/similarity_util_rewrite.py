from sympy.strategies.tools import top_down
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import numpy as np
from typing import Union

class finalOutputAnalysis:
    def __init__(self, model_name: str, device: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)
        self.max_logit_indices = None
        self.max_logit_values = None

        self.max_cosine_indices = None
        self.max_cosine_values = None

        self.model_head_weights = self.model.lm_head.weight.data
        self.model_head_weights_normalized = self.model_head_weights / self.model_head_weights.norm(dim=-1, keepdim=True)
        self.random_vectors = F.normalize(torch.randn(self.model_head_weights.shape[0], self.model_head_weights.shape[1]), dim=-1).to(device)
        self.intersection_count = 0
        

    def pass_input(self, input_text: Union[str, torch.Tensor], topk: int =10) -> None:
        """
        Passing input through the model
        creates max logit indices, values, max cosine indices, values, and max random vector cosine similarities
        
        Args:
            input_text: Either a string to be tokenized or a tensor of token IDs
            topk: Number of top-k results to compute
        """

        
        if isinstance(input_text, str):
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        elif isinstance(input_text, torch.Tensor):
            # If tensor, assume it's token IDs and format as dictionary
            if input_text.dim() == 1:
                input_text = input_text.unsqueeze(0)  # Add batch dimension if needed
            inputs = {"input_ids": input_text.to(self.model.device)}
        else:
            raise TypeError(f"input_text must be str or torch.Tensor, got {type(input_text)}")
        
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)
        last_output = outputs.hidden_states[-1]
        last_output_normalized = last_output / last_output.norm(dim=-1, keepdim=True)
        # print(last_output.shape)
        # print(outputs.logits)

        ### largest logits and their indices
        self.max_logit_values, self.max_logit_indices = torch.topk(outputs.logits, k=topk, dim=-1)

        ### largest cosine similarity
        cosine_similarities = last_output_normalized @ self.model_head_weights_normalized.T
        self.max_cosine_values, self.max_cosine_indices = torch.topk(cosine_similarities, k=topk, dim=-1)

        # print(self.max_logit_values)
        # print(self.max_cosine_values)

        ### with random indices
        random_similarities = last_output_normalized @ self.random_vectors.T
        max_random_cosine_values, _ = torch.topk(random_similarities, k=topk, dim=-1)

        #print(max_random_cosine_values)
    
    def find_intersect_numbers(self):
        """
        Finds the number of topk indices that appear on both cosine and logit.
        """
        #print(self.max_cosine_indices.shape) # shape Batch X seq_len X topk
        #print(self.max_logit_indices.shape)
        intersection_counts = []
        
        # Let's do the dumb way first.
        for i in range(self.max_cosine_indices.shape[-2]): # seq_len
            mc = self.max_cosine_indices[0][i].cpu().numpy()
            ml = self.max_logit_indices[0][i].cpu().numpy()
            mc = set(mc)
            ml = set(ml)
            intersection = mc & ml
            intersection_count = len(intersection)
            intersection_counts.append(intersection_count)
        self.intersection_count = intersection_counts







if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.2-1B"
    device = "mps"
    random_string = "What a wonderful world"
    foa = finalOutputAnalysis(model_name, device)
    foa.pass_input(random_string)
    foa.find_intersect_numbers()
    print(foa.intersection_count)
    print(foa.max_cosine_indices)

    del foa
    torch.mps.empty_cache()

            


