using System;
using System.Linq;

namespace TransformerLib.Layers
{
    /// <summary>
    /// 层归一化层
    /// </summary>
    public class LayerNormalization : ILayer
    {
        private readonly int _embeddingDim;
        private readonly float[] _gamma;
        private readonly float[] _beta;
        private float[] _normalizedInput;
        private float _mean;
        private float _variance;

        /// <summary>
        /// 构造函数，初始化层归一化层
        /// </summary>
        /// <param name="embeddingDim">嵌入维度</param>
        public LayerNormalization(int embeddingDim)
        {
            _normalizedInput = new float[0];
            _embeddingDim = embeddingDim;
            _gamma = new float[embeddingDim];
            _beta = new float[embeddingDim];
            InitializeParameters();
        }

        /// <summary>
        /// 初始化参数
        /// </summary>
        private void InitializeParameters()
        {
            for (int i = 0; i < _embeddingDim; i++)
            {
                _gamma[i] = 1.0f;
                _beta[i] = 0.0f;
            }
        }

        /// <summary>
        /// 计算均值和方差
        /// </summary>
        /// <param name="input">输入数据</param>
        /// <returns>返回均值和方差</returns>
        private (float mean, float variance) CalculateMeanVariance(float[] input)
        {
            float mean = input.Average();
            float variance = input.Select(x => (x - mean) * (x - mean)).Average();
            return (mean, variance);
        }

        /// <summary>
        /// 前向传播
        /// </summary>
        /// <param name="input">输入数据</param>
        /// <returns>返回归一化后的向量</returns>
        public float[] Forward(float[] input)
        {
            _normalizedInput = new float[input.Length];
            (_mean, _variance) = CalculateMeanVariance(input);
            for (int i = 0; i < input.Length; i++)
            {
                _normalizedInput[i] = (input[i] - _mean) / (float)Math.Sqrt(_variance + 1e-5);
                _normalizedInput[i] = (_gamma[i] * _normalizedInput[i]) + _beta[i];
            }
            return _normalizedInput;
        }

        /// <summary>
        /// 反向传播
        /// </summary>
        /// <param name="gradient">输入梯度</param>
        /// <returns>返回梯度</returns>
        public float[] Backward(float[] gradient)
        {
            float[] inputGradient = new float[gradient.Length];
            float gradMean = 0;
            float gradVariance = 0;

            for (int i = 0; i < gradient.Length; i++)
            {
                gradMean += gradient[i];
                gradVariance += gradient[i] * (inputGradient[i] - _mean);
            }

            for (int i = 0; i < gradient.Length; i++)
            {
                inputGradient[i] = gradient[i] - (gradMean / gradient.Length) - ((inputGradient[i] - _mean) * gradVariance / (gradient.Length * _variance));
                inputGradient[i] = _gamma[i] * inputGradient[i] / (float)Math.Sqrt(_variance + 1e-5);
            }

            return inputGradient;
        }

        /// <summary>
        /// 更新权重
        /// </summary>
        /// <param name="learningRate">学习率</param>
        public void UpdateWeights(float learningRate)
        {
            for (int i = 0; i < _gamma.Length; i++)
            {
                _gamma[i] -= learningRate * _gamma[i];
                _beta[i] -= learningRate * _beta[i];
            }
        }
    }
}
