using System;

namespace TransformerLib.Layers
{
    /// <summary>
    /// 嵌入层
    /// </summary>
    public class Embedding : ILayer
    {
        private readonly int _vocabSize;
        private readonly int _embeddingDim;
        private readonly float[,] _weights;
        private float[,] _inputGradient;

        /// <summary>
        /// 构造函数，初始化嵌入层
        /// </summary>
        /// <param name="vocabSize">词汇表大小</param>
        /// <param name="embeddingDim">嵌入维度</param>
        public Embedding(int vocabSize, int embeddingDim)
        {
            _inputGradient = new float[0, 0];
            _vocabSize = vocabSize;
            _embeddingDim = embeddingDim;
            _weights = new float[vocabSize, embeddingDim];
            InitializeWeights();
        }

        /// <summary>
        /// 初始化权重
        /// </summary>
        private void InitializeWeights()
        {
            Random rand = new Random();
            for (int i = 0; i < _vocabSize; i++)
            {
                for (int j = 0; j < _embeddingDim; j++)
                {
                    _weights[i, j] = (float)rand.NextDouble();
                }
            }
        }

        /// <summary>
        /// 前向传播
        /// </summary>
        /// <param name="input">输入数据</param>
        /// <returns>返回嵌入向量</returns>
        public float[] Forward(float[] input)
        {
            _inputGradient = new float[input.Length, _embeddingDim];
            float[] result = new float[input.Length * _embeddingDim];
            for (int i = 0; i < input.Length; i++)
            {
                int index = (int)input[i];
                for (int j = 0; j < _embeddingDim; j++)
                {
                    result[(i * _embeddingDim) + j] = _weights[index, j];
                    _inputGradient[i, j] = _weights[index, j];
                }
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
            float[] inputGradient = new float[_vocabSize];
            for (int i = 0; i < gradient.Length / _embeddingDim; i++)
            {
                for (int j = 0; j < _embeddingDim; j++)
                {
                    int index = (int)_inputGradient[i, j];
                    inputGradient[index] += gradient[(i * _embeddingDim) + j];
                }
            }
            return inputGradient;
        }

        /// <summary>
        /// 更新权重
        /// </summary>
        /// <param name="learningRate">学习率</param>
        public void UpdateWeights(float learningRate)
        {
            for (int i = 0; i < _vocabSize; i++)
            {
                for (int j = 0; j < _embeddingDim; j++)
                {
                    _weights[i, j] -= learningRate * _inputGradient[i, j];
                }
            }
        }
    }
}
