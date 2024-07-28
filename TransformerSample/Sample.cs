using TransformerLib.Data;
using TransformerLib.Models;

namespace TransformerSample
{
    public class Sample
    {
        public static void Predict()
        {
            // 数据加载
            IDataLoader dataLoader = new TextDataLoader();
            IEnumerable<string> data = dataLoader.LoadData("data.txt");

            // 初始化词汇表和分词器
            string[] vocabularys = File.ReadAllLines("vocab.txt");
            Vocabulary vocabulary = new Vocabulary(vocabularys);
            Tokenizer tokenizer = new Tokenizer(vocabulary);

            // 转换数据
            List<int> tokenizedData = new List<int>();
            foreach (string sentence in data)
            {
                tokenizedData.AddRange(tokenizer.Tokenize(sentence));
            }

            // 初始化Transformer模型
            IModel transformer = new Transformer(vocabSize: 10000, embeddingDim: 512 * 10, maxLen: 128, numHeads: 8, hiddenDim: 2048);
            transformer.Train(data);

            // 进行预测
            string input = "this is a sample input";
            string prediction = transformer.Predict(input);
            Console.WriteLine(prediction);
        }
    }
}
