using System.Collections.Generic;
using System.IO;

namespace TransformerLib.Data
{
    /// <summary>
    /// 文本数据加载器，实现IDataLoader接口
    /// </summary>
    public class TextDataLoader : IDataLoader
    {
        public IEnumerable<string> LoadData(string path)
        {
            return File.ReadAllLines(path);
        }
    }
}
