using System.Collections.Generic;

namespace TransformerLib.Data
{
    /// <summary>
    /// 文本分词器
    /// </summary>
    public class Tokenizer
    {
        private readonly Vocabulary _vocabulary;

        /// <summary>
        /// 构造函数，初始化词汇表
        /// </summary>
        /// <param name="vocabulary">词汇表</param>
        public Tokenizer(Vocabulary vocabulary)
        {
            _vocabulary = vocabulary;
        }

        /// <summary>
        /// 将文本转换为标记
        /// </summary>
        /// <param name="text">输入文本</param>
        /// <returns>返回标记列表</returns>
        public List<int> Tokenize(string text)
        {
            List<int> tokens = new List<int>();
            string[] words = text.Split(' ');

            foreach (string? word in words)
            {
                tokens.Add(_vocabulary.GetTokenId(word));
            }

            return tokens;
        }

        /// <summary>
        /// 将标记转换为文本
        /// </summary>
        /// <param name="tokens">标记列表</param>
        /// <returns>返回文本</returns>
        public string Detokenize(List<int> tokens)
        {
            List<string?> words = new List<string?>();

            foreach (int token in tokens)
            {
                if (_vocabulary.GetWord(token) is string vt)
                {
                    words.Add(vt);
                }
                else
                {
                    words.Add("[UNK]");
                }
            }

            return string.Join(" ", words);
        }
    }
}
