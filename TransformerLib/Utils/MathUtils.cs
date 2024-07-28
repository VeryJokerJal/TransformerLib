using System;
using System.Linq;

namespace TransformerLib.Utils
{
    /// <summary>
    /// 数学工具类
    /// </summary>
    public static class MathUtils
    {
        /// <summary>
        /// 计算向量的点积
        /// </summary>
        /// <param name="vector1">向量1</param>
        /// <param name="vector2">向量2</param>
        /// <returns>返回点积结果</returns>
        public static float DotProduct(float[] vector1, float[] vector2)
        {
            if (vector1.Length != vector2.Length)
            {
                throw new ArgumentException("向量维度不匹配");
            }

            float result = 0;
            for (int i = 0; i < vector1.Length; i++)
            {
                result += vector1[i] * vector2[i];
            }
            return result;
        }

        /// <summary>
        /// 计算矩阵乘法
        /// </summary>
        /// <param name="matrix1">矩阵1</param>
        /// <param name="matrix2">矩阵2</param>
        /// <returns>返回乘法结果矩阵</returns>
        public static float[,] MatrixMultiply(float[,] matrix1, float[,] matrix2)
        {
            int rows1 = matrix1.GetLength(0);
            int cols1 = matrix2.GetLength(0);
            int rows2 = matrix2.GetLength(0);
            int cols2 = matrix2.GetLength(1);

            if (cols1 != rows2)
            {
                throw new ArgumentException("矩阵维度不匹配");
            }

            float[,] result = new float[rows1, cols2];

            for (int i = 0; i < rows1; i++)
            {
                for (int j = 0; j < cols2; j++)
                {
                    for (int k = 0; k < cols1; k++)
                    {
                        result[i, j] += matrix1[i, k] * matrix2[k, j];
                    }
                }
            }

            return result;
        }

        /// <summary>
        /// 计算矩阵与向量的乘法
        /// </summary>
        /// <param name="matrix">矩阵</param>
        /// <param name="vector">向量</param>
        /// <returns>返回乘法结果向量</returns>
        public static float[,] MatrixMultiply(float[] vector, float[,] matrix)
        {
            int rows = 1;
            int cols = vector.Length;

            if (cols != matrix.GetLength(0))
            {
                throw new ArgumentException("矩阵和向量维度不匹配");
            }

            float[,] result = new float[rows, matrix.GetLength(1)];

            for (int j = 0; j < matrix.GetLength(1); j++)
            {
                for (int i = 0; i < cols; i++)
                {
                    result[0, j] += vector[i] * matrix[i, j];
                }
            }

            return result;
        }

        /// <summary>
        /// 转置矩阵
        /// </summary>
        /// <param name="matrix">输入矩阵</param>
        /// <returns>返回转置后的矩阵</returns>
        public static float[,] Transpose(float[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            float[,] result = new float[cols, rows];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[j, i] = matrix[i, j];
                }
            }

            return result;
        }

        /// <summary>
        /// 计算矩阵的Softmax
        /// </summary>
        /// <param name="matrix">输入矩阵</param>
        /// <returns>返回Softmax后的矩阵</returns>
        public static float[,] Softmax(float[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            float[,] result = new float[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                float maxVal = matrix[i, 0];
                for (int j = 1; j < cols; j++)
                {
                    if (matrix[i, j] > maxVal)
                    {
                        maxVal = matrix[i, j];
                    }
                }

                float sumExp = 0;
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = (float)Math.Exp(matrix[i, j] - maxVal);
                    sumExp += result[i, j];
                }

                for (int j = 0; j < cols; j++)
                {
                    result[i, j] /= sumExp;
                }
            }

            return result;
        }

        /// <summary>
        /// 计算向量的Softmax
        /// </summary>
        /// <param name="values">输入值数组</param>
        /// <returns>返回Softmax后的数组</returns>
        public static float[] Softmax(float[] values)
        {
            float maxVal = values.Max();
            float sumExp = 0;
            float[] expValues = new float[values.Length];

            for (int i = 0; i < values.Length; i++)
            {
                expValues[i] = (float)Math.Exp(values[i] - maxVal);
                sumExp += expValues[i];
            }

            for (int i = 0; i < values.Length; i++)
            {
                expValues[i] /= sumExp;
            }

            return expValues;
        }

        /// <summary>
        /// 计算ReLU激活函数
        /// </summary>
        /// <param name="values">输入值数组</param>
        /// <returns>返回ReLU后的数组</returns>
        public static float[] ReLU(float[] values)
        {
            float[] result = new float[values.Length];
            for (int i = 0; i < values.Length; i++)
            {
                result[i] = Math.Max(0, values[i]);
            }
            return result;
        }

        /// <summary>
        /// 计算矩阵的ReLU激活函数
        /// </summary>
        /// <param name="matrix">输入矩阵</param>
        /// <returns>返回ReLU后的矩阵</returns>
        public static float[,] ReLU(float[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            float[,] result = new float[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = Math.Max(0, matrix[i, j]);
                }
            }

            return result;
        }

        /// <summary>
        /// 缩放矩阵
        /// </summary>
        /// <param name="matrix">输入矩阵</param>
        /// <param name="scale">缩放因子</param>
        /// <returns>返回缩放后的矩阵</returns>
        public static float[,] ScaleMatrix(float[,] matrix, float scale)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            float[,] result = new float[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = matrix[i, j] * scale;
                }
            }

            return result;
        }

        /// <summary>
        /// 扁平化矩阵
        /// </summary>
        /// <param name="matrix">输入矩阵</param>
        /// <returns>返回扁平化后的数组</returns>
        public static float[] Flatten(float[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            float[] result = new float[rows * cols];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[(i * cols) + j] = matrix[i, j];
                }
            }

            return result;
        }

        /// <summary>
        /// 将向量转换为矩阵
        /// </summary>
        /// <param name="vector">输入向量</param>
        /// <returns>返回转换后的矩阵</returns>
        public static float[,] VectorToMatrix(float[] vector)
        {
            float[,] matrix = new float[1, vector.Length];
            for (int i = 0; i < vector.Length; i++)
            {
                matrix[0, i] = vector[i];
            }
            return matrix;
        }

        /// <summary>
        /// 将矩阵转换为向量
        /// </summary>
        /// <param name="matrix">输入矩阵</param>
        /// <returns>返回转换后的向量</returns>
        public static float[] MatrixToVector(float[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            float[] vector = new float[rows * cols];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    vector[(i * cols) + j] = matrix[i, j];
                }
            }
            return vector;
        }
    }
}
