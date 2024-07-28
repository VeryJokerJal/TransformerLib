using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TransformerLib.Data;
using TransformerLib.Layers;
using TransformerLib.Utils;

namespace TransformerLib.Models
{
    /// <summary>
    /// Transformer模型
    /// </summary>
    public class Transformer : IModel
    {
        private readonly Embedding _embedding;
        private readonly PositionalEncoding _positionalEncoding;
        private readonly Encoder _encoder;
        private readonly Decoder _decoder;
        private readonly int _maxLen;
        private readonly float _learningRate = 0.001f;

        /// <summary>
        /// 构造函数，初始化Transformer模型
        /// </summary>
        /// <param name="vocabSize">词汇表大小</param>
        /// <param name="embeddingDim">嵌入维度</param>
        /// <param name="maxLen">最大长度</param>
        /// <param name="numHeads">头数</param>
        /// <param name="hiddenDim">隐藏层维度</param>
        public Transformer(int vocabSize, int embeddingDim, int maxLen, int numHeads, int hiddenDim)
        {
            _embedding = new Embedding(vocabSize, embeddingDim);
            _positionalEncoding = new PositionalEncoding(maxLen, embeddingDim);
            _encoder = new Encoder(numHeads, embeddingDim, hiddenDim);
            _decoder = new Decoder(numHeads, embeddingDim, hiddenDim);
            _maxLen = maxLen;
        }

        /// <summary>
        /// 训练模型
        /// </summary>
        /// <param name="data">训练数据</param>
        public void Train(IEnumerable<string> data)
        {
            string[] vocabularys = File.ReadAllLines("vocab.txt");
            Tokenizer tokenizer = new Tokenizer(new Vocabulary(vocabularys));
            List<int[]> tokenizedData = data.Select(sentence => tokenizer.Tokenize(sentence).ToArray()).ToList();

            for (int epoch = 0; epoch < 100; epoch++)
            {
                float totalLoss = 0;
                foreach (int[]? tokens in tokenizedData)
                {
                    int[] inputTokens = tokens.Take(_maxLen).ToArray();
                    int[] targetTokens = tokens.Skip(1).Take(_maxLen).ToArray();

                    // 前向传播
                    float[] inputEmbedding = _embedding.Forward(inputTokens.Select(t => (float)t).ToArray());
                    float[] inputPositioned = _positionalEncoding.Forward(inputEmbedding);
                    float[] encoderOutput = _encoder.Forward(inputPositioned);

                    float[] targetEmbedding = _embedding.Forward(targetTokens.Select(t => (float)t).ToArray());
                    float[] targetPositioned = _positionalEncoding.Forward(targetEmbedding);
                    _decoder.SetEncoderOutput(encoderOutput);
                    float[] decoderOutput = _decoder.Forward(targetPositioned);

                    // 计算损失
                    float[] outputSoftmax = MathUtils.Softmax(decoderOutput);
                    float loss = ComputeLoss(outputSoftmax, targetTokens);
                    totalLoss += loss;

                    // 反向传播
                    float[] lossGradient = ComputeLossGradient(outputSoftmax, targetTokens);
                    float[] decoderGradient = _decoder.Backward(lossGradient);
                    float[] encoderGradient = _encoder.Backward(decoderGradient);

                    // 更新参数
                    UpdateParameters();
                }
                System.Console.WriteLine($"Epoch {epoch + 1}, Loss: {totalLoss / tokenizedData.Count}");
            }
        }

        /// <summary>
        /// 计算损失
        /// </summary>
        /// <param name="predictions">预测值</param>
        /// <param name="targets">目标值</param>
        /// <returns>返回损失值</returns>
        private float ComputeLoss(float[] predictions, int[] targets)
        {
            float loss = 0;
            for (int i = 0; i < targets.Length; i++)
            {
                loss -= (float)Math.Log(predictions[targets[i]]);
            }
            return loss / targets.Length;
        }

        /// <summary>
        /// 计算损失梯度
        /// </summary>
        /// <param name="predictions">预测值</param>
        /// <param name="targets">目标值</param>
        /// <returns>返回损失梯度</returns>
        private float[] ComputeLossGradient(float[] predictions, int[] targets)
        {
            float[] gradient = new float[predictions.Length];
            for (int i = 0; i < targets.Length; i++)
            {
                gradient[targets[i]] = -1.0f / predictions[targets[i]];
            }
            return gradient;
        }

        /// <summary>
        /// 更新模型参数
        /// </summary>
        private void UpdateParameters()
        {
            _embedding.UpdateWeights(_learningRate);
            _encoder.UpdateWeights(_learningRate);
            _decoder.UpdateWeights(_learningRate);
        }

        /// <summary>
        /// 进行预测
        /// </summary>
        /// <param name="input">输入数据</param>
        /// <returns>返回预测结果</returns>
        public string Predict(string input)
        {
            // 将输入文本转换为标记
            string[] vocabularys = File.ReadAllLines("vocab.txt");
            Tokenizer tokenizer = new Tokenizer(new Vocabulary(vocabularys));
            int[] tokens = tokenizer.Tokenize(input).ToArray();

            // 嵌入和位置编码
            float[] embeddedInput = _embedding.Forward(tokens.Select(t => (float)t).ToArray());
            float[] encodedInput = _positionalEncoding.Forward(embeddedInput);

            // 编码器前向传播
            float[] encoderOutput = _encoder.Forward(encodedInput);

            // 解码器前向传播
            _decoder.SetEncoderOutput(encoderOutput);
            float[] decoderOutput = _decoder.Forward(encoderOutput);

            // 将输出标记转换为文本
            List<int> outputTokens = decoderOutput.Select(d => (int)d).ToList();
            return tokenizer.Detokenize(outputTokens);
        }
    }
}
