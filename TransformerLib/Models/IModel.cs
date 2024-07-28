using System.Collections.Generic;

namespace TransformerLib.Models
{
    /// <summary>
    /// 定义模型接口
    /// </summary>
    public interface IModel
    {
        /// <summary>
        /// 训练模型
        /// </summary>
        /// <param name="data">训练数据</param>
        void Train(IEnumerable<string> data);

        /// <summary>
        /// 进行预测
        /// </summary>
        /// <param name="input">输入数据</param>
        /// <returns>返回预测结果</returns>
        string Predict(string input);
    }
}
