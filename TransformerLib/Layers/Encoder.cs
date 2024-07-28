namespace TransformerLib.Layers
{
    /// <summary>
    /// 编码器层
    /// </summary>
    public class Encoder : ILayer
    {
        private readonly MultiHeadAttention _multiHeadAttention;
        private readonly FeedForward _feedForward;
        private readonly LayerNormalization _layerNormalization1;
        private readonly LayerNormalization _layerNormalization2;
        private float[] _attentionOutput;
        private float[] _normalizedAttention;
        private float[] _feedForwardOutput;
        private float[] _normalizedOutput;

        /// <summary>
        /// 构造函数，初始化编码器层
        /// </summary>
        /// <param name="numHeads">头数</param>
        /// <param name="embeddingDim">嵌入维度</param>
        /// <param name="hiddenDim">隐藏层维度</param>
        public Encoder(int numHeads, int embeddingDim, int hiddenDim)
        {
            _attentionOutput = new float[0];
            _normalizedAttention = new float[0];
            _feedForwardOutput = new float[0];
            _normalizedOutput = new float[0];

            _multiHeadAttention = new MultiHeadAttention(numHeads, embeddingDim);
            _feedForward = new FeedForward(embeddingDim, hiddenDim);
            _layerNormalization1 = new LayerNormalization(embeddingDim);
            _layerNormalization2 = new LayerNormalization(embeddingDim);
        }

        /// <summary>
        /// 前向传播
        /// </summary>
        /// <param name="input">输入数据</param>
        /// <returns>返回编码后的向量</returns>
        public float[] Forward(float[] input)
        {
            _attentionOutput = _multiHeadAttention.Forward(input);
            _normalizedAttention = _layerNormalization1.Forward(_attentionOutput);

            _feedForwardOutput = _feedForward.Forward(_normalizedAttention);
            _normalizedOutput = _layerNormalization2.Forward(_feedForwardOutput);

            return _normalizedOutput;
        }

        /// <summary>
        /// 反向传播
        /// </summary>
        /// <param name="gradient">输入梯度</param>
        /// <returns>返回梯度</returns>
        public float[] Backward(float[] gradient)
        {
            float[] normalizedGradient2 = _layerNormalization2.Backward(gradient);
            float[] feedForwardGradient = _feedForward.Backward(normalizedGradient2);

            float[] normalizedGradient1 = _layerNormalization1.Backward(feedForwardGradient);
            float[] attentionGradient = _multiHeadAttention.Backward(normalizedGradient1);

            return attentionGradient;
        }

        /// <summary>
        /// 更新权重
        /// </summary>
        /// <param name="learningRate">学习率</param>
        public void UpdateWeights(float learningRate)
        {
            _multiHeadAttention.UpdateWeights(learningRate);
            _feedForward.UpdateWeights(learningRate);
            _layerNormalization1.UpdateWeights(learningRate);
            _layerNormalization2.UpdateWeights(learningRate);
        }
    }
}
