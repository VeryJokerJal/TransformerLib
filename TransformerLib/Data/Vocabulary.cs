using System;
using System.Collections.Generic;
using System.IO;

namespace TransformerLib.Data
{
    /// <summary>
    /// 词汇表类
    /// </summary>
    public class Vocabulary
    {
        public List<string> Words { get; }

        private readonly Dictionary<string, int> _wordToId;
        private readonly Dictionary<int, string> _idToWord;

        /// <summary>
        /// 构造函数，初始化词汇表
        /// </summary>
        /// <param name="words">词汇列表</param>
        public Vocabulary(IEnumerable<string> words)
        {
            Words = new List<string>();
            _wordToId = new Dictionary<string, int>();
            _idToWord = new Dictionary<int, string>();

            int id = 0;
            foreach (string word in words)
            {
                Words.Add(word);
                _wordToId[word] = id;
                _idToWord[id] = word;
                id++;
            }
        }

        public void LoadFromFile(string filePath)
        {
            try
            {
                using StreamReader reader = new StreamReader(filePath);
                string line;
                while ((line = reader.ReadLine()) != null)
                {
                    Words.Add(line);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error loading vocabulary: {ex.Message}");
            }
        }

        /// <summary>
        /// 获取单词对应的标记ID
        /// </summary>
        /// <param name="word">单词</param>
        /// <returns>返回标记ID</returns>
        public int GetTokenId(string word)
        {
            return _wordToId.TryGetValue(word, out int id) ? id : Words.IndexOf("[UNK]");
        }

        /// <summary>
        /// 获取标记ID对应的单词
        /// </summary>
        /// <param name="id">标记ID</param>
        /// <returns>返回单词</returns>
        public string? GetWord(int id)
        {
            return _idToWord.TryGetValue(id, out string? word) ? word : null;
        }
    }
}
