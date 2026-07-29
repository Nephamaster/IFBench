# Generalizing Verifiable Instruction Following

This repo contains IFBench, which is a new, challenging benchmark for precise instruction following.
Read the <a href="https://arxiv.org/pdf/2507.02833">IFBench paper</a>, accepted to NeurIPS 2025, D&B.

## IFBench
IFBench consists of two parts:

- OOD Constraints: 58 new and challenging constraints, with corresponding verification functions. The constraint templates are combined with prompts from a held-out set of WildChat (Zhao et al. 2024).

- (optionally) Multiturn Constraint Isolation in 2 turns: The prompt and the constraint are separated over two turns, i.e. the first turn is the user prompt and the model's response to the prompt, and the second turn is the constraint that modifies the initial prompt.

- New IF-RLVR training constraints: 29 new and challenging constraints, with corresponding verification functions. 

## How to run the evaluation
Install the requirements via the requirements.txt file.
You need two jsonl files, one the IFBench_test.jsonl file (in the data folder) and one your file with eval prompts and completions (see sample_output.jsonl as an example). Then run:
```
python3 -m run_eval --input_data=IFBench_test.jsonl --input_response_data=sample_output.jsonl --output_dir=eval
```

Note: In the paper we generally report the prompt-level loose accuracy of IFBench. When we generate for evaluation, we use a temperature of 0 and adjust the maximum generated tokens depending on the model type, i.e. for thinking models we allow to generate more tokens and we then process the output to extract the answer without the reasoning chains.

## Released Datasets
You can find our released datasets in this [collection](https://huggingface.co/collections/allenai/ifbench-683f590687f61b512558cdf1), which contains the [test data](https://huggingface.co/datasets/allenai/IFBench_test), the [multi-turn test data](https://huggingface.co/datasets/allenai/IFBench_multi-turn) and the [IF-RLVR training data](https://huggingface.co/datasets/allenai/IF_multi_constraints_upto5).

## RLVR for Precise Instruction Following
We also release our IF-RLVR code, as part of [open-instruct](https://github.com/allenai/open-instruct). You can run this [GRPO script](https://github.com/allenai/open-instruct/blob/main/open_instruct/grpo_fast.py), using our [training data](https://huggingface.co/datasets/allenai/IF_multi_constraints_upto5). This is an [example command](https://github.com/allenai/open-instruct/blob/main/scripts/train/rlvr/valpy_if_grpo_fast.sh).

The new training constraints and verification functions are here: https://github.com/allenai/open-instruct/tree/main/open_instruct/IFEvalG

## 📊 Model Performance Leaderboard

| Rank | Model | IFBench Score | IFEval Score |
|------|-------|---------------|--------------|
| 🥇 1 | OpenAI o3 | **69.3** | 95.0 |
| 🥈 2 | Qwen2.5 Base + IF-RLVR | **53.7** | 87.8 |
| 🥉 3 |  Llama 3.1 Base + IF-RLVR | **52.7** | 88.2 |
| 4 | Gemini 2.5 Pro | 52.3 | 65.4 |
| 5 | Qwen 2.5 Instruct + IF-RLVR | 48.7 | 89.1 |
| 6 | OLMo2 Base + IF-RLVR | 47.3 | 70.4 |
| 7 | OLMo2 Instruct + IF-RLVR | 44.7 | 74.5 |
| 7 | Tulu3 DPO + IF-RLVR | 43.3 | 92.2 |
| 9 | Claude 4 Sonnet | 42.3 | 91.3 |
| 10 | DeepSeek R1 | 38.0 | 86.13 |
| 11 | Qwen 3 32B | 37.3 | 85.6 |
| 12 | Qwen 3 8B | 35.0 | 86.3 |

*Sorted by IFBench score (higher is better)*
If you want your model added to the leaderboard, please create a pull request or email me!

## Licensing

This codebase is licensed under Apache 2.0 as given in [LICENSE](./LICENSE).

The data is licensed under ODC-BY-1.0. It is intended for research and educational use in accordance with Ai2's Responsible Use Guidelines. The dataset includes output data generated from third party models that are subject to separate terms governing their use.


## Acknowledgements

Parts of IFBench are built upon and extend [IFEval](https://github.com/google-research/google-research/tree/master/instruction_following_eval) (Zhou et al. 2023) and we would like to thank them for their great work!


## Citation

If you used this repository or our models, please cite our work:

```bibtex
@misc{pyatkin2025generalizing,
   title={Generalizing Verifiable Instruction Following}, 
   author={Valentina Pyatkin and Saumya Malik and Victoria Graf and Hamish Ivison and Shengyi Huang and Pradeep Dasigi and Nathan Lambert and Hannaneh Hajishirzi},
   year={2025},
  journal={Advances in Neural Information Processing Systems},
  volume={38},
  year={2025}
}
```

## Local vLLM inference for Qwen3 and Llama-3.1

The repository includes `serve_vllm.sh`, `inference.sh`, and
`generate_responses.py` for evaluating locally merged models through vLLM.
The evaluation configuration follows the paper setting:

- `temperature=0.0` (greedy decoding);
- `repetition_penalty=1.0`;
- Qwen3 non-thinking prompt construction;
- an explicit chat-turn stop token for each backbone.

The explicit stop token is required because the token used to end an assistant
turn is not necessarily the tokenizer's ordinary EOS token:

| Backbone | Assistant-turn token | Token ID |
|---|---|---:|
| Qwen3 | `<|im_end|>` | `151645` |
| Llama-3.1 | `<|eot_id|>` | `128009` |

Start one model server:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=8002 \
  bash serve_vllm.sh qwen3 Kmeans 01
```

Run generation and evaluation in another shell:

```bash
PORT=8002 WORKERS=32 bash inference.sh qwen3 Kmeans 01
```

Llama-3.1 uses the corresponding model kind:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=8002 \
  bash serve_vllm.sh llama31 Kmeans 01

PORT=8002 WORKERS=32 bash inference.sh llama31 Kmeans 01
```

The generated JSONL retains the fields required by IFBench (`prompt` and
`response`) and also records `finish_reason`, `stop_reason`, and token usage.
A large number of `finish_reason="length"` records means generation reached
`max_tokens`; inspect those responses for repetition or incorrect stop-token
handling. Do not resume files generated with the old non-stopping setup unless
they have been cleaned first.
