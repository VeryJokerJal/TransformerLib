using System;

namespace TransformerLib.Layers
{
    /// <summary>
    /// 位置编码层
    /// </summary>
    public class PositionalEncoding : ILayer
    {
        private readonly int _maxLen;
        private readonly int _embeddingDim;
        private readonly float[,] _positionalEncoding;

        /// <summary>
        /// 构造函数，初始化位置编码层
        /// </summary>
        /// <param name="maxLen">最大长度</param>
        /// <param name="embeddingDim">嵌入维度</param>
        public PositionalEncoding(int maxLen, int embeddingDim)
        {
            _maxLen = maxLen;
            _embeddingDim = embeddingDim;
            _positionalEncoding = new float[maxLen, embeddingDim];
            CalculatePositionalEncoding();
        }

        /// <summary>
        /// 计算位置编码
        /// </summary>
        private void CalculatePositionalEncoding()
        {
            for (int pos = 0; pos < _maxLen; pos++)
            {
                for (int i = 0; i < _embeddingDim; i++)
                {
                    _positionalEncoding[pos, i] = i % 2 == 0
                        ? (float)Math.Sin(pos / Math.Pow(10000, 2.0 * i / _embeddingDim))
                        : (float)Math.Cos(pos / Math.Pow(10000, 2.0 * (i - 1) / _embeddingDim));
                }
            }
        }

        /// <summary>
        /// 前向传播
        /// </summary>
        /// <param name="input">输入数据</param>
        /// <returns>返回位置编码后的向量</returns>
        public float[] Forward(float[] input)
        {
            float[] result = new float[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                int pos = i / _embeddingDim;
                int dim = i % _embeddingDim;
                result[i] = input[i] + _positionalEncoding[pos, dim];
            }
            return result;
        }

        /// <summary>
        /// 反向传播
        /// </summary>
        /// <param name="gradient">输入梯度</param>
        /// <returns>返回梯度</returns>
        public float[] Backward(float[] gradient)
        {
            // 对于位置编码层，反向传播梯度与输入梯度相同
            return gradient;
        }

        public void UpdateWeights(float learningRate)
        {

        }
    }
}
