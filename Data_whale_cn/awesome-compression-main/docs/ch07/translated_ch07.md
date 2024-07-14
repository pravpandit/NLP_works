# Chapter 7 Project Practice

<!-- This chapter will integrate the model compression methods introduced earlier through a comprehensive project practice to help learners better understand -->

&emsp;&emsp;The model compression algorithms mentioned in this tutorial, pruning, quantization, distillation, and neural network architecture search, are mutually perpendicular in practical applications, and each has its own applicable scenarios and trade-offs. However, we can combine multiple model compression algorithms to achieve better results.

&emsp;&emsp;At present, the combination of model compression algorithms with good results, rich research and application mainly includes:
- Combination of pruning and quantization
- Combination of knowledge distillation and quantization
- Combination of pruning and knowledge distillation
- Combination of neural architecture search and knowledge distillation

&emsp;&emsp;In addition, some recent studies have also proposed some more comprehensive combination methods of model compression algorithms, such as:
- Combination of knowledge distillation with pruning and quantization
- Combination of neural network architecture search with pruning and quantization

&emsp;&emsp;This tutorial will combine practical applications, for common image classification tasks, by applying different compression methods to the ResNet-18 model, and compare from the perspective of hardware efficiency improvement, model capability retention (considering the difficulty of implementation, mainly based on the CIFAR-10 dataset), and algorithm implementation difficulty

<!-- In addition, two versions of the code, cpu-only and cuda, are provided -->

<!-- TODO: Add the practice section of LLM model compression -->

## 7.1 Combining pruning and quantization

&emsp;&emsp;Both pruning and quantization can effectively reduce the redundancy in model weights and activation values, thereby reducing the size and computation of the model.

&emsp;&emsp;The previous chapter mentioned:

>Pruning (from the timing) can be divided into:
> - Pruning after training (static sparsity)
> - Pruning during training (dynamic sparsity)
> - Pruning before training

>And quantization is mainly divided into:
> - Post-training quantization PTQ
> - Quantization-aware training QAT

&emsp;&emsp;For the deployment and compression acceleration of existing models, the common solution is to combine post-training pruning and post-training quantization. In addition, for large language models LLM, QLoRA can also be used during iterative pruning to greatly reduce the memory overhead and data requirements of fine-tuning.

### 7.1.1 Preliminary attempt (simple combination)
&emsp;&emsp;Simply and intuitively, the methods of combining pruning and quantization can be:
&emsp;&emsp;**Prune first, then quantize**: First prune the model, then quantize the pruned model.
&emsp;&emsp;**Quantize first, then prune**: First quantize the model, then prune the quantized model.

&emsp;&emsp;However, for such a simpleA series of problems will arise when piecing together the models:
For **Prune first, then quantize**
- The pruned model reduces the number of information channels, resulting in reduced robustness of the model, and thus reduced tolerance to noise introduced by quantization, resulting in reduced performance of the quantized model.
- Pruning usually sets the weights to zero, resulting in sparsity. However, quantization can map these zero values ​​to non-zero quantized values, thereby reducing sparsity. In particular, if fine-tuning is performed in quantization and quantization-aware training (QAT) is performed after pruning, some of these zero values ​​will also be adjusted to non-zero values ​​during training (if there is no special treatment), thereby reducing sparsity.

For **Quantize first, then prune**
- Since it is more difficult to perform backpropagation on quantized neural networks, pruning methods with better performance (iterative pruning, etc.) are almost inseparable from retraining (fine-tuning, rewinding...).

&emsp;&emsp;Therefore, we need to conduct a deeper study on the combination of pruning and quantization to solve these problems.

### 7.1.2 Problem Analysis and Solution

Frederick Tung & Greg Mori (2018) in the paper [CLIP-Q: Deep Network Compression Learning by In-Parallel Pruning-Quantization](https://doi.org/10.1109/cvpr.2018.00821) proposed to combine pruning and quantization in a learning framework (Bayesian optimization), combining pruning and quantization through partitioning, and fine-tuning in parallel. (The pruning in this paper adopts unstructured pruning, which is more concerned with the model size. The quantization in this paper is not directly quantized to integers, but the mean of the partition intervals)

Shaokai Ye et.al (2018) proposed a unified framework of weight pruning and clustering/quantization based on ADMM in the paper [A Unified Framework of DNN Weight Pruning and Weight Clustering/Quantization Using ADMM](https://arxiv.org/abs/1811.01907), and used the alternating direction multiplier method (ADMM) to convert the non-convex optimization problem of minimizing the model error into sub-problems of minimizing the error caused by weight changes and the error caused by bias changes, and effectively solve them.

Haichuan Yang et.al (2020) proposed a unified framework of weight pruning and clustering/quantization based on ADMM in the paper [Automatic Neural Network Compression by Sparsity-Quantization Joint Learning: A Constrained Optimization-based ApproaIn [Bayesian Bits: Unifying Quantization and Pruning](https://arxiv.org/abs/1910.05897), a method is proposed to automatically jointly prune and quantize DNNs according to the target model size based on unified pruning and quantization using ADMM, without using any hyperparameters to manually set the compression ratio of each layer.

Mart van Baalen et.al (2020) proposed a gradient-optimized pruning and mixed-precision quantization method in the paper [Bayesian Bits: Unifying Quantization and Pruning](https://arxiv.org/abs/2005.07093), which decomposes the quantization problem into a series of bit width doubling problems. At each new bit width, the residual between the full-precision value and the previous rounded value will be quantized. Then it is decided whether to add this quantization residual to obtain a higher effective bit width and lower quantization noise. Because starting from a power-of-2 bit width, this decomposition will always produce a hardware-friendly configuration and serve as a unified view of pruning and quantization through an additional 0-bit option.

Gil Shomron et.al (2021) in the paper [Post-Training Sparsity-Aware Quantization](https://arxiv.org/abs/2105.11010) proposed a method for efficient quantization of the pruned model. First, the weights and activations are quantized to 8 bits by skipping zero values. Then, the paired activation values ​​are analyzed to see if there are zeros. If yes, 8 bits are retained. If not, all are quantized to 4 bits, thus achieving a slight loss of accuracy and practical hardware performance. (The pruning in this paper is mainly unstructured and dynamically sparse)

Shipeng Bai et.al (2023) proposed a new framework called unified data-free compression (UDFC) in the paper [Unified Data-Free Compression: Pruning and Quantization without Fine-Tuning](https://arxiv.org/abs/2308.07209), which can perform pruning and quantization simultaneously without any data and fine-tuning process. First, it is assumed that part of the information of the damaged (e.g., pruned or quantized) channel can be preserved by linear combinations of other channels, and then the reconstruction form is derived from the assumption of recovering the information lost due to compression. Finally, we formulate the reconstruction error between the original network and its compressed network, and theoretically derive the closed-form solution.

&emsp;&emsp;