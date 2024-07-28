# Transformer 模型库

## 项目简介

本项目实现了一个简化版的 Transformer 模型库，包括嵌入层、位置编码、编码器和解码器等核心组件。通过此库，用户可以进行文本的嵌入、编码和解码操作，适用于自然语言处理中的序列到序列任务，如机器翻译、文本生成等。

## 项目原理

Transformer 模型是 Vaswani 等人在 2017 年提出的一种基于注意力机制的序列到序列模型。与传统的 RNN 和 LSTM 不同，Transformer 模型完全基于注意力机制，无需按顺序处理输入数据，因此能够更好地捕捉全局信息，并行化处理速度更快。

### Transformer 的基本结构

Transformer 由编码器（Encoder）和解码器（Decoder）两个主要部分组成，每个部分又由若干个层堆叠而成。编码器将输入序列转换为隐藏表示，解码器根据隐藏表示生成输出序列。

#### 编码器（Encoder）

编码器由多个相同的层（Layer）堆叠而成。每个层包括两个子层：
1. 多头自注意力机制（Multi-Head Self-Attention）
2. 前馈神经网络（Feed Forward Neural Network）

每个子层后面都跟有一个残差连接和层归一化（Layer Normalization）。

##### 多头自注意力机制（Multi-Head Self-Attention）

多头自注意力机制允许模型关注输入序列的不同部分，捕捉序列中的各种依赖关系。其基本步骤如下：

1. **计算查询（Query）、键（Key）和值（Value）**：输入通过线性变换生成查询、键和值。
   \[
   Q = XW_Q, \quad K = XW_K, \quad V = XW_V
   \]
   其中，\(W_Q, W_K, W_V\) 为参数矩阵。

2. **计算注意力权重**：通过缩放点积计算查询和键的相似度，并应用 Softmax 函数。
   \[
   \text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
   \]
   其中，\(d_k\) 为键的维度。

3. **多头注意力**：将输入分为多个头，分别计算注意力，最后拼接结果。
   \[
   \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W_O
   \]
   其中，每个头的计算为：
   \[
   \text{head}_i = \text{Attention}(QW_{Q_i}, KW_{K_i}, VW_{V_i})
   \]

##### 前馈神经网络（Feed Forward Neural Network）

前馈神经网络由两个线性变换层和一个激活函数（通常为 ReLU）组成。
\[
\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
\]

#### 解码器（Decoder）

解码器的结构与编码器类似，但在每个层中增加了一个编码器-解码器注意力机制，用于结合编码器的输出。每个解码器层包括三个子层：

1. **多头自注意力机制**
2. **编码器-解码器注意力机制**
3. **前馈神经网络**

### 残差连接和层归一化

在每个子层之后，使用残差连接和层归一化：
\[
\text{Output} = \text{LayerNorm}(x + \text{SubLayer}(x))
\]

### 位置编码（Positional Encoding）

由于 Transformer 模型不含有顺序信息，需要通过位置编码为每个词添加位置信息。位置编码使用正弦和余弦函数生成：
\[
\text{PE}_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{\frac{2i}{d_{\text{model}}}}}\right)
\]
\[
\text{PE}_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{\frac{2i}{d_{\text{model}}}}}\right)
\]

### 整体模型架构

整个 Transformer 模型架构如下：

1. **输入嵌入**：输入序列经过嵌入层和位置编码后，输入到编码器。
2. **编码器**：编码器将嵌入序列转换为隐藏表示。
3. **解码器**：解码器根据隐藏表示生成输出序列。
4. **输出层**：解码器输出经过线性变换和 Softmax 函数生成最终预测。

## 代码结构

项目代码结构如下：

```plaintext
TransformerLib/
├── Data/
│   ├── DataLoader.cs
│   ├── Tokenizer.cs
│   └── Vocabulary.cs
├── Layers/
│   ├── Embedding.cs
│   ├── PositionalEncoding.cs
│   ├── MultiHeadAttention.cs
│   ├── FeedForward.cs
│   ├── LayerNormalization.cs
│   ├── Encoder.cs
│   └── Decoder.cs
├── Models/
│   └── Transformer.cs
├── Utils/
│   └── MathUtils.cs
└── TransformerApp/
    └── Program.cs
