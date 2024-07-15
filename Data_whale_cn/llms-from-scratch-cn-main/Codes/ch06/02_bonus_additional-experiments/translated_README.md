# Additional experiments

The following table adds some experiments to answer additional questions about various design choices. The first row uses the same settings as the main section and is used as a reference.
For example,

- Comparing row 1 and row 2 answers the question: "What is the performance difference when we train on the last or first token?";
- Comparing row 1 and row 3 answers the question: "What is the performance difference when we train only the last layer instead of the last block?";
- And so on.

&nbsp;

| | Model | Weights | Trainable token | Trainable layers | Context length | Training acc | Validation acc | Test acc | Training time | CPU/GPU |
| ---- | ------------------ | ---------- | --------------- | ---------------- | ----------------------- | ------------ | -------------- | -------- | ------------- | ------- |
| 1 | gpt2-small (124M) | pretrained | last | last_block | longest train ex. (120) | 96.63% | 99.33% | 95.00% | 0.28 min | A100 |
| 2 | gpt2-small (124M) | pretrained | first | last_block | longest train ex. (120) | 78.46% | 80.54% | 75.00% | 0.28 min | A100 |
| 3 | gpt2-small (124M) | pretrained | first | last_block | longest train ex. (120) | 78.46% | 80.54% | 75.00% | 0.28 min | A100 |
| 3 | gpt2-small (124M) | pretrainined | last | last_layer | longest train ex. (120) | 78.65% | 79.87% | 72.00% | 0.25 min | A100 |
| 4 | gpt2-small (124M) | pretrained | last | last_two_blocks | longest train ex. (120) | 98.85% | 98.66% | 98.33% | 0.33 min | A100 |
| 5 | gpt2-small (124M) | pretrained | last | all | longest train ex.(120) | 99.62% | 96.64% | 96.67% | 0.69 min | A100 |
| 6 | gpt2-medium (355M) | pretrained | last | last_block | longest train ex. (120) | 87.50% | 91.28% | 84.67% | 0.75 min | A100 |
| 7 | gpt2-large (774M) | pretrained | last | last_block | longest train ex. (120) | 99.52% | 98.66%| 96.67% | 1.50 min | A100 |
| 8 | gpt2-xl (1558M) | pretrained | last | last_block | longest train ex. (120) | 99.81% | 99.33% | 98.33% | 2.83 min | A100 |
| 9 | gpt2-small (124M) | random | last | all | longest train ex. (120) | 100% | 96.64% | 93.67% | 0.69 min | A100 |
| 10 | gpt2-small (124M) | pretrained | last | LoRA | longest train ex. (120) | 100.00% | 97.32% | 96.67% | 0.75 min | A100 |
| 11 | gpt2-small (124M) | pretrained | last | last_block | context length (1024) | 83.08% | 87.92% | 78.33% | 2.46 min | A100 |
| 12 | gpt2-small (124M) | pretrained | last | last_block | variable: nopadding (batch size 1) | 100.00% | 98.66% | 98.00% | 1.75 min | A100 |
| 13 | gpt2-small (124M) | pretrained | last | last_block | variable: no padding (batch size 8) | 99.33% | 98.66% | 98.33% | 1.70 min | A100 |
| 14 | gpt2-small (124M) | pretrained | last | last_block | longest train ex. (120); but no causal mask | 99.23% | 98.66%| 95.33% | 0.29 min | A100 |
| 15 | gpt2-small (124M) | pretrained | last | last_block | longest train ex. (120) and `ignore_index` for padding | 96.63% | 99.33% | 95.00% | 0.28 min | A100 |

&nbsp;

## Usage

You can use the following code to reproduce the experiment:

- Row 1: `python additional-experiments.py`
- Row 2: `python additional-experiments.py --trainable_token_pos first`
- Row 3: `python additional-experiments.py --trainable_layers last_layer`
- Row 4: `pythonn additional-experiments.py --trainable_layers last_two_blocks`
- Row 5: `python additional-experiments.py --trainable_layers all`
- Row 6: `python additional-experiments.py --model_size "gpt2-medium (355M)"`
- Row 7: `python additional-experiments.py --model_size "gpt2-large (774M)"`
- Row 8: `python additional-experiments.py --model_size "gpt2-xl (1558M)"`
- Row 9: `python additional-experiments.py --weights random --trainable_layers all`
- Row 10: `python additional-experiments.py --trainable_layers lora --lora_rank 16 --lora_alpha 16`
- Row 11: `python additional-experiments.py --context_length "model_context_length"`
- Row 12: `python additional-experiments.py --no_padding --batch_size 1`
- Row 13: `python additional-experiments.py --no_padding --batch_size 1 --accumulation_steps 8`
- Row 14: `python additional-experiments.py --disable_causal_mask`
- Row 15: `python additional-experiments.py --ignore_index 50256`

I intentionally kept the LLM and the dataset small, so if you don't have access to a GPU, you can run it in about 10 seconds on a regular laptop like a MacBook Air M3.15 minutes of training.

&nbsp;

## Explanation
1. **Training the last output token vs. the first output token (row 1 vs. row 2)**: Training the last output token leads to better performance compared to the first output token. This improvement is expected due to the causal self-attention mask.
2. **Training the last Transformer block vs. the last layer (row 1 vs. row 3)**: Training the entire last Transformer block also leads to better results than training only the last layer.
3. **Training all layers vs. the last Transformer block (row 1 vs. row 4)**: Training all layers shows a modest improvement of about 2% over training only the last Transformer block, but it requires almost three times as much training time.
4. **Training the last Transformer block with all layers (row 1 vs. row 5)**: Training all layers shows a modest improvement of about 2% over training only the last Transformer block, but it takes almost three times as long to train in terms of time. Also, it only trains the last two of the 12 Transformer blocks, which also perform poorly.
5. **Using a larger pre-trained model (row 1 vs. row 5, and row 1 vs. rows 7 and 8)**: Using a 3x larger pre-trained model leads to worse results. However, as expected, using a 5x larger model improves performance compared to the initial model. Similarly, a 12x larger model further improves prediction performance. (The medium model may not have been pre-trained very well, or a particular fine-tuning configuration does not work well for that model.)
6. **Using a model with random weights vs. pre-trained weights (row 1 vs. row 9)**: Using a model with random weights produces results that are only slightly worse than using pre-trained weights by 1.3%.
7. **Using LoRA (low-order adaptation) vs. training all layers (row 10 vs. row 5)**: Keeping the model frozen and adding a trainable LoRA layer is a viable alternative to training all model parameters (see [Appendix E](../../appendix-E/01_main-chapter-code/appendix-E.ipynb)), and can even improve performance by 1%. This is likely due to less overfitting, as can be seen from the ~1% reduction in the gap between training and validation accuracy when using LoRA. It is also slightly faster using LoRA because fewer parameters need to be updated.
8. **Padding input to full context length vs. longest training example (row 1 vs. row 11)**: Padding the input to the full supported context length gives significantly worse results.
9.**Padding vs. No Padding (Lines 1 vs. 12 and 13)**: The `--no_padding` option disables padding in the dataset, which requires training the model with a batch size of 1 since the inputs have variable length. This results in better test accuracy but requires longer training time. In Line 12, we additionally enable gradient accumulation for 8 steps to achieve the same batch size as the other experiments, which helps reduce overfitting and slightly improves test set accuracy.
10. **Disable Causal Attention Mask (Lines 1 vs. 14)**: Disable the causal attention mask used in the multi-head attention module. This means that all tokens can attend to all other tokens. The model accuracy is slightly improved compared to the GPT model with the causal mask.
11. **Ignore padding index in loss and backpropagation (lines 1 and 15)**: Setting `--ignore_index 50256` excludes the `|endoftext|` padding token in the `cross_entropy` loss function in PyTorch. In this case, it has no effect because we replace the output layer so that the token IDs for binary classification examples are either 0 or 1. However, this setting is useful when fine-tuning the model using the instructions in Chapter 7.