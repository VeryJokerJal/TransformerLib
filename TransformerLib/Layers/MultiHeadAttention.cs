using System;
using TransformerLib.Utils;

namespace TransformerLib.Layers
{
    /// <summary>
    /// 多头注意力层
    /// </summary>
    public class MultiHeadAttention : ILayer
    {
        private readonly int _numHeads;
        private readonly int _embeddingDim;
        private readonly float[,] _weightsQ;
        private readonly float[,] _weightsK;
        private readonly float[,] _weightsV;
        private readonly float[,] _weightsO;
        private float[,] _Q;
        private float[,] _K;
        private float[,] _V;
        private float[,] _attention;

        /// <summary>
        /// 构造函数，初始化多头注意力层
        /// </summary>
        /// <param name="numHeads">头数</param>
        /// <param name="embeddingDim">嵌入维度</param>
        public MultiHeadAttention(int numHeads, int embeddingDim)
        {
            _Q = new float[0, 0];
            _K = new float[0, 0];
            _V = new float[0, 0];
            _attention = new float[0, 0];

            _numHeads = numHeads;
            _embeddingDim = embeddingDim;

            _weightsQ = new float[embeddingDim, embeddingDim];
            _weightsK = new float[embeddingDim, embeddingDim];
            _weightsV = new float[embeddingDim, embeddingDim];
            _weightsO = new float[embeddingDim, embeddingDim];

            InitializeWeights();
        }

        /// <summary>
        /// 初始化权重
        /// </summary>
        private void InitializeWeights()
        {
            Random rand = new Random();
            for (int i = 0; i < _embeddingDim; i++)
            {
                for (int j = 0; j < _embeddingDim; j++)
                {
                    _weightsQ[i, j] = (float)rand.NextDouble();
                    _weightsK[i, j] = (float)rand.NextDouble();
                    _weightsV[i, j] = (float)rand.NextDouble();
                    _weightsO[i, j] = (float)rand.NextDouble();
                }
            }
        }

        /// <summary>
        /// 计算注意力权重
        /// </summary>
        /// <param name="Q">查询矩阵</param>
        /// <param name="K">键矩阵</param>
        /// <param name="V">值矩阵</param>
        /// <returns>返回注意力输出</returns>
        private float[,] CalculateAttention(float[,] Q, float[,] K, float[,] V)
        {
            float[,] scores = MathUtils.MatrixMultiply(Q, MathUtils.Transpose(K));
            float[,] scaledScores = MathUtils.ScaleMatrix(scores, 1.0f / (float)Math.Sqrt(_embeddingDim));
            float[,] softmaxScores = MathUtils.Softmax(scaledScores);

            return MathUtils.MatrixMultiply(softmaxScores, V);
        }

        /// <summary>
        /// 前向传播（自注意力）
        /// </summary>
        /// <param name="input">输入数据</param>
        /// <returns>返回注意力后的向量</returns>
        public float[] Forward(float[] input)
        {
            float[,] inputMatrix = MathUtils.VectorToMatrix(input);

            _Q = MathUtils.MatrixMultiply(inputMatrix, _weightsQ);
            _K = MathUtils.MatrixMultiply(inputMatrix, _weightsK);
            _V = MathUtils.MatrixMultiply(inputMatrix, _weightsV);

            _attention = CalculateAttention(_Q, _K, _V);

            float[,] result = MathUtils.MatrixMultiply(_attention, _weightsO);
            return MathUtils.Flatten(result);
        }

        /// <summary>
        /// 前向传播（编码器-解码器注意力）
        /// </summary>
        /// <param name="input">输入数据</param>
        /// <param name="encoderOutput">编码器输出</param>
        /// <returns>返回注意力后的向量</returns>
        public float[] Forward(float[] input, float[] encoderOutput)
        {
            float[,] inputMatrix = MathUtils.VectorToMatrix(input);
            float[,] encoderOutputMatrix = MathUtils.VectorToMatrix(encoderOutput);

            _Q = MathUtils.MatrixMultiply(inputMatrix, _weightsQ);
            _K = MathUtils.MatrixMultiply(encoderOutputMatrix, _weightsK);
            _V = MathUtils.MatrixMultiply(encoderOutputMatrix, _weightsV);

            _attention = CalculateAttention(_Q, _K, _V);

            float[,] result = MathUtils.MatrixMultiply(_attention, _weightsO);
            return MathUtils.Flatten(result);
        }

        /// <summary>
        /// 反向传播
        /// </summary>
        /// <param name="gradient">输入梯度</param>
        /// <returns>返回梯度</returns>
        public float[] Backward(float[] gradient)
        {
            float[,] gradMatrix = MathUtils.VectorToMatrix(gradient);

            // 计算 O 的梯度
            float[,] gradO = MathUtils.MatrixMultiply(MathUtils.Transpose(_attention), gradMatrix);

            // 计算注意力的梯度
            float[,] gradAttention = MathUtils.MatrixMultiply(gradMatrix, MathUtils.Transpose(_weightsO));

            // 计算 Q, K, V 的梯度
            float[,] gradQ = MathUtils.MatrixMultiply(gradAttention, MathUtils.Transpose(_K));
            float[,] gradK = MathUtils.MatrixMultiply(MathUtils.Transpose(gradAttention), _Q);
            float[,] gradV = MathUtils.MatrixMultiply(gradAttention, _V);

            // 更新权重
            for (int i = 0; i < _weightsQ.GetLength(0); i++)
            {
                for (int j = 0; j < _weightsQ.GetLength(1); j++)
                {
                    _weightsQ[i, j] -= gradQ[i, j];
                    _weightsK[i, j] -= gradK[i, j];
                    _weightsV[i, j] -= gradV[i, j];
                    _weightsO[i, j] -= gradO[i, j];
                }
            }

            return MathUtils.Flatten(gradQ);
        }

        /// <summary>
        /// 更新权重
        /// </summary>
        /// <param name="learningRate">学习率</param>
        public void UpdateWeights(float learningRate)
        {
            for (int i = 0; i < _weightsQ.GetLength(0); i++)
            {
                for (int j = 0; j < _weightsQ.GetLength(1); j++)
                {
                    _weightsQ[i, j] -= learningRate * _weightsQ[i, j];
                    _weightsK[i, j] -= learningRate * _weightsK[i, j];
                    _weightsV[i, j] -= learningRate * _weightsV[i, j];
                    _weightsO[i, j] -= learningRate * _weightsO[i, j];
                }
            }
        }
    }
}
