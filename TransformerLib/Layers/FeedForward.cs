using System;
using TransformerLib.Utils;

namespace TransformerLib.Layers
{
    /// <summary>
    /// 前馈神经网络层
    /// </summary>
    public class FeedForward : ILayer
    {
        private readonly int _inputDim;
        private readonly int _hiddenDim;
        private readonly float[,] _weights1;
        private readonly float[,] _weights2;
        private float[,] _input;
        private float[,] _hidden;

        /// <summary>
        /// 构造函数，初始化前馈神经网络层
        /// </summary>
        /// <param name="inputDim">输入维度</param>
        /// <param name="hiddenDim">隐藏层维度</param>
        public FeedForward(int inputDim, int hiddenDim)
        {
            _input = new float[0, 0];
            _hidden = new float[0, 0];

            _inputDim = inputDim;
            _hiddenDim = hiddenDim;

            _weights1 = new float[inputDim, hiddenDim];
            _weights2 = new float[hiddenDim, inputDim];

            InitializeWeights();
        }

        /// <summary>
        /// 初始化权重
        /// </summary>
        private void InitializeWeights()
        {
            Random rand = new Random();
            for (int i = 0; i < _inputDim; i++)
            {
                for (int j = 0; j < _hiddenDim; j++)
                {
                    _weights1[i, j] = (float)rand.NextDouble();
                }
            }
            for (int i = 0; i < _hiddenDim; i++)
            {
                for (int j = 0; j < _inputDim; j++)
                {
                    _weights2[i, j] = (float)rand.NextDouble();
                }
            }
        }

        /// <summary>
        /// 前向传播
        /// </summary>
        /// <param name="input">输入数据</param>
        /// <returns>返回前馈后的向量</returns>
        public float[] Forward(float[] input)
        {
            _input = MathUtils.VectorToMatrix(input);

            _hidden = MathUtils.MatrixMultiply(_input, _weights1);
            _hidden = MathUtils.ReLU(_hidden);
            float[,] output = MathUtils.MatrixMultiply(_hidden, _weights2);

            return MathUtils.Flatten(output);
        }

        /// <summary>
        /// 反向传播
        /// </summary>
        /// <param name="gradient">输入梯度</param>
        /// <returns>返回梯度</returns>
        public float[] Backward(float[] gradient)
        {
            float[,] gradMatrix = MathUtils.VectorToMatrix(gradient);

            // 计算权重2的梯度
            float[,] gradWeights2 = MathUtils.MatrixMultiply(MathUtils.Transpose(_hidden), gradMatrix);

            // 计算隐藏层的梯度
            float[,] gradHidden = MathUtils.MatrixMultiply(gradMatrix, MathUtils.Transpose(_weights2));

            // 计算ReLU的梯度
            for (int i = 0; i < _hidden.GetLength(0); i++)
            {
                for (int j = 0; j < _hidden.GetLength(1); j++)
                {
                    if (_hidden[i, j] <= 0)
                    {
                        gradHidden[i, j] = 0;
                    }
                }
            }

            // 计算权重1的梯度
            float[,] gradWeights1 = MathUtils.MatrixMultiply(MathUtils.Transpose(_input), gradHidden);

            // 更新权重
            for (int i = 0; i < _weights1.GetLength(0); i++)
            {
                for (int j = 0; j < _weights1.GetLength(1); j++)
                {
                    _weights1[i, j] -= gradWeights1[i, j];
                }
            }
            for (int i = 0; i < _weights2.GetLength(0); i++)
            {
                for (int j = 0; j < _weights2.GetLength(1); j++)
                {
                    _weights2[i, j] -= gradWeights2[i, j];
                }
            }

            return MathUtils.Flatten(gradHidden);
        }

        /// <summary>
        /// 更新权重
        /// </summary>
        /// <param name="learningRate">学习率</param>
        public void UpdateWeights(float learningRate)
        {
            for (int i = 0; i < _weights1.GetLength(0); i++)
            {
                for (int j = 0; j < _weights1.GetLength(1); j++)
                {
                    _weights1[i, j] -= learningRate * _weights1[i, j];
                }
            }
            for (int i = 0; i < _weights2.GetLength(0); i++)
            {
                for (int j = 0; j < _weights2.GetLength(1); j++)
                {
                    _weights2[i, j] -= learningRate * _weights2[i, j];
                }
            }
        }
    }
}
