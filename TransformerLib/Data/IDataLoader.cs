using System.Collections.Generic;

namespace TransformerLib.Data
{
    /// <summary>
    /// 定义数据加载接口
    /// </summary>
    public interface IDataLoader
    {
        /// <summary>
        /// 加载数据的方法
        /// </summary>
        /// <param name="path">数据文件路径</param>
        /// <returns>返回数据枚举集合</returns>
        IEnumerable<string> LoadData(string path);
    }
}
