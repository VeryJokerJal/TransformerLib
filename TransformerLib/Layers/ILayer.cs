namespace TransformerLib.Layers
{
    /// <summary>
    /// 定义层接口
    /// </summary>
    public interface ILayer
    {
        /// <summary>
        /// 前向传播
        /// </summary>
        /// <param name="input">输入数据</param>
        /// <returns>返回输出数据</returns>
        float[] Forward(float[] input);

        /// <summary>
        /// 反向传播
        /// </summary>
        /// <param name="gradient">输入梯度</param>
        /// <returns>返回梯度</returns>
        float[] Backward(float[] gradient);

        /// <summary>
        /// 更新权重
        /// </summary>
        /// <param name="learningRate">学习率</param>
        void UpdateWeights(float learningRate);
    }
}
