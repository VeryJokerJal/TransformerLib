namespace TransformerLib.Layers
{
    /// <summary>
    /// 解码器层
    /// </summary>
    public class Decoder : ILayer
    {
        private readonly MultiHeadAttention _selfAttention;
        private readonly MultiHeadAttention _encoderDecoderAttention;
        private readonly FeedForward _feedForward;
        private readonly LayerNormalization _layerNormalization1;
        private readonly LayerNormalization _layerNormalization2;
        private readonly LayerNormalization _layerNormalization3;
        private float[] _selfAttentionOutput;
        private float[] _normalizedSelfAttention;
        private float[] _encoderDecoderAttentionOutput;
        private float[] _normalizedEncoderDecoderAttention;
        private float[] _feedForwardOutput;
        private float[] _normalizedOutput;
        private float[] _encoderOutput;

        /// <summary>
        /// 构造函数，初始化解码器层
        /// </summary>
        /// <param name="numHeads">头数</param>
        /// <param name="embeddingDim">嵌入维度</param>
        /// <param name="hiddenDim">隐藏层维度</param>
        public Decoder(int numHeads, int embeddingDim, int hiddenDim)
        {
            _selfAttentionOutput = new float[0];
            _normalizedSelfAttention = new float[0];
            _encoderDecoderAttentionOutput = new float[0];
            _normalizedEncoderDecoderAttention = new float[0];
            _feedForwardOutput = new float[0];
            _normalizedOutput = new float[0];
            _encoderOutput = new float[0];

            _selfAttention = new MultiHeadAttention(numHeads, embeddingDim);
            _encoderDecoderAttention = new MultiHeadAttention(numHeads, embeddingDim);
            _feedForward = new FeedForward(embeddingDim, hiddenDim);
            _layerNormalization1 = new LayerNormalization(embeddingDim);
            _layerNormalization2 = new LayerNormalization(embeddingDim);
            _layerNormalization3 = new LayerNormalization(embeddingDim);
        }

        /// <summary>
        /// 设置编码器输出
        /// </summary>
        /// <param name="encoderOutput">编码器输出</param>
        public void SetEncoderOutput(float[] encoderOutput)
        {
            _encoderOutput = encoderOutput;
        }

        /// <summary>
        /// 前向传播
        /// </summary>
        /// <param name="input">输入数据</param>
        /// <returns>返回解码后的向量</returns>
        public float[] Forward(float[] input)
        {
            _selfAttentionOutput = _selfAttention.Forward(input);
            _normalizedSelfAttention = _layerNormalization1.Forward(_selfAttentionOutput);

            _encoderDecoderAttentionOutput = _encoderDecoderAttention.Forward(_normalizedSelfAttention, _encoderOutput);
            _normalizedEncoderDecoderAttention = _layerNormalization2.Forward(_encoderDecoderAttentionOutput);

            _feedForwardOutput = _feedForward.Forward(_normalizedEncoderDecoderAttention);
            _normalizedOutput = _layerNormalization3.Forward(_feedForwardOutput);

            return _normalizedOutput;
        }

        /// <summary>
        /// 反向传播
        /// </summary>
        /// <param name="gradient">输入梯度</param>
        /// <returns>返回梯度</returns>
        public float[] Backward(float[] gradient)
        {
            float[] normalizedGradient3 = _layerNormalization3.Backward(gradient);
            float[] feedForwardGradient = _feedForward.Backward(normalizedGradient3);

            float[] normalizedGradient2 = _layerNormalization2.Backward(feedForwardGradient);
            float[] encoderDecoderAttentionGradient = _encoderDecoderAttention.Backward(normalizedGradient2);

            float[] normalizedGradient1 = _layerNormalization1.Backward(encoderDecoderAttentionGradient);
            float[] selfAttentionGradient = _selfAttention.Backward(normalizedGradient1);

            return selfAttentionGradient;
        }

        /// <summary>
        /// 更新权重
        /// </summary>
        /// <param name="learningRate">学习率</param>
        public void UpdateWeights(float learningRate)
        {
            _selfAttention.UpdateWeights(learningRate);
            _encoderDecoderAttention.UpdateWeights(learningRate);
            _feedForward.UpdateWeights(learningRate);
            _layerNormalization1.UpdateWeights(learningRate);
            _layerNormalization2.UpdateWeights(learningRate);
            _layerNormalization3.UpdateWeights(learningRate);
        }
    }
}
