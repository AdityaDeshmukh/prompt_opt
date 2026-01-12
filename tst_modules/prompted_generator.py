from transformers import AutoTokenizer, pipeline
from tqdm import tqdm
from typing import Optional, List


class PromptedGenerator:
    def __init__(
        self,
        model: str,
        template: str,
        end_punct: str,
        pad_token: str,
        device_id: int,
        lower_outputs: bool,
        control_output_length: bool,
        backend: str = "transformers"
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model,
                                                       pad_token=pad_token)
        self.backend = backend
        self.generator = None
        self.vllm_model = None
        if backend == "vllm":
            try:
                from vllm import LLM  # type: ignore
            except ImportError:
                print("vLLM not installed; falling back to transformers pipeline.")
                self.backend = "transformers"
            else:
                self.vllm_model = LLM(model=model)
        if self.backend == "transformers":
            self.generator = pipeline("text-generation",
                                      model=model,
                                      tokenizer=self.tokenizer,
                                      device=device_id)
        
        self.template = template
        self.end_punct = end_punct
        self.lower_outputs = lower_outputs
        self.control_output_length = control_output_length

    def _get_max_new_tokens(
        self,
        seq_len: int,
        control_output_length: Optional[bool] = None
    ) -> Optional[int]:
        if control_output_length is None:
            control_output_length = self.control_output_length
        if control_output_length:
            # This hack tends to speed up generation compared to default
            return max(1.5 * seq_len, seq_len + 10)
        else:
            return None

    def sample_generate(
        self,
        prompt: str,
        source_text: str,
        num_samples: int,
        top_k: Optional[int],
        top_p: float,
        lower_outputs: Optional[bool] = None,
        control_output_length: Optional[bool] = None,
        **kwargs
    ) -> List[str]:
        # Used for controlling output length
        formatted_template = self.template.format(prompt=prompt,
                                                  sentence_1=source_text)

        src_len = len(self.tokenizer(source_text)['input_ids'])
        max_new_tokens = self._get_max_new_tokens(
            src_len, control_output_length=control_output_length)
        pad_token_id = self.tokenizer.pad_token_id

        if self.backend == "vllm":
            return self._sample_generate_vllm(
                formatted_template,
                num_samples,
                top_k,
                top_p,
                max_new_tokens,
            )

        sample_outputs = self.generator(formatted_template,
                                        pad_token_id=pad_token_id,
                                        do_sample=True,
                                        max_new_tokens=max_new_tokens,
                                        top_k=top_k,
                                        top_p=top_p,
                                        num_return_sequences=num_samples,
                                        # Only return generated text, without the prompt
                                        return_full_text=False,
                                        **kwargs)
        generated_texts = []
        for output in sample_outputs:
            text = output["generated_text"]
            generated_texts.append(self.postprocess_output(text))
        return generated_texts

    def sample_generate_batch(
        self,
        prompt: str,
        source_texts: List[str],
        num_samples: int,
        top_k: Optional[int],
        top_p: float,
        lower_outputs: Optional[bool] = None,
        control_output_length: Optional[bool] = None,
        **kwargs
    ) -> List[List[str]]:
        all_generated_texts = []
        if self.backend == "vllm":
            formatted_prompts = [
                self.template.format(prompt=prompt, sentence_1=source_text)
                for source_text in source_texts
            ]
            src_lengths = [
                len(self.tokenizer(source_text)['input_ids'])
                for source_text in source_texts
            ]
            max_new_tokens = [
                self._get_max_new_tokens(
                    src_len, control_output_length=control_output_length
                )
                for src_len in src_lengths
            ]
            max_new_tokens = max(
                [token_count for token_count in max_new_tokens if token_count],
                default=None,
            )
            return self._sample_generate_vllm_batch(
                formatted_prompts,
                num_samples,
                top_k,
                top_p,
                max_new_tokens,
            )
        for i, source_text in tqdm(enumerate(source_texts),
                                   total=len(source_texts)):
            generated_texts = self.sample_generate(
                prompt, source_text, num_samples, top_k, top_p,
                lower_outputs=lower_outputs,
                control_output_length=control_output_length)
            all_generated_texts.append(generated_texts)
        return all_generated_texts

    def _sample_generate_vllm(
        self,
        formatted_prompt: str,
        num_samples: int,
        top_k: Optional[int],
        top_p: float,
        max_new_tokens: Optional[int],
    ) -> List[str]:
        if self.vllm_model is None:
            raise RuntimeError("vLLM backend requested but not initialized.")
        from vllm import SamplingParams  # type: ignore

        params = SamplingParams(
            n=num_samples,
            top_k=top_k if top_k is not None else -1,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )
        outputs = self.vllm_model.generate([formatted_prompt], params)
        generated_texts = []
        for output in outputs[0].outputs:
            generated_texts.append(self.postprocess_output(output.text))
        return generated_texts

    def _sample_generate_vllm_batch(
        self,
        formatted_prompts: List[str],
        num_samples: int,
        top_k: Optional[int],
        top_p: float,
        max_new_tokens: Optional[int],
    ) -> List[List[str]]:
        if self.vllm_model is None:
            raise RuntimeError("vLLM backend requested but not initialized.")
        from vllm import SamplingParams  # type: ignore

        params = SamplingParams(
            n=num_samples,
            top_k=top_k if top_k is not None else -1,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )
        outputs = self.vllm_model.generate(formatted_prompts, params)
        all_generated_texts = []
        for output in outputs:
            all_generated_texts.append(
                [self.postprocess_output(sample.text) for sample in output.outputs]
            )
        return all_generated_texts

    def postprocess_output(
        self,
        text: str,
        end_punct: Optional[str] = None,
        lower_outputs: Optional[bool] = None
    ) -> str:
        if end_punct is None:
            end_punct = self.end_punct
        if lower_outputs is None:
            lower_outputs = self.lower_outputs

        try:
            end = text.index(end_punct)
        except ValueError:
            end = len(text)
        text = text[:end].strip()

        try:
            end = text.index('.')
        except ValueError:
            end = len(text)
        try:
            end = min(end, text.index('!'))
        except ValueError:
            end = end
        try:
            end = min(end, text.index('?'))
        except ValueError:
            end = end

        text = text[:end+1].strip()
        if lower_outputs:
            text = text.lower()

        return text
