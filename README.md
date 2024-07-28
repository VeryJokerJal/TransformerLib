# Transformer 模型库

## 项目简介

本项目实现了一个简化版的 Transformer 模型库，包括嵌入层、位置编码、编码器和解码器等核心组件。通过此库，用户可以进行文本的嵌入、编码和解码操作，适用于自然语言处理中的序列到序列任务，如机器翻译、文本生成等。

## 项目原理

Transformer 模型是 Vaswani 等人在 2017 年提出的一种基于注意力机制的序列到序列模型。与传统的 RNN 和 LSTM 不同，Transformer 模型完全基于注意力机制，无需按顺序处理输入数据，因此能够更好地捕捉全局信息，并行化处理速度更快。

### 主要组件

1. **嵌入层（Embedding Layer）**：
   - 将输入的词索引转换为对应的词向量。
   - 使用随机初始化的权重矩阵进行词嵌入。

2. **位置编码（Positional Encoding）**：
   - 为每个词向量添加位置信息，使模型能够识别词序。
   - 使用正弦和余弦函数生成位置编码。

3. **多头注意力机制（Multi-Head Attention）**：
   - 通过多个注意力头来捕捉不同子空间的信息。
   - 使用缩放点积注意力（Scaled Dot-Product Attention）计算注意力权重。

4. **前馈神经网络（Feed Forward Neural Network）**：
   - 在每个注意力层后添加前馈神经网络，用于进一步处理特征。

5. **编码器（Encoder）**：
   - 由多个编码器层堆叠而成，每层包括多头注意力机制和前馈神经网络。

6. **解码器（Decoder）**：
   - 结构与编码器类似，但每个解码器层中增加了编码器-解码器注意力机制，用于结合编码器的输出。

## 代码结构

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
