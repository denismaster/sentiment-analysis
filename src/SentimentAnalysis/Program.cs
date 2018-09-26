using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System;
using System.IO;
using System.Threading.Tasks;

namespace SentimentAnalysis
{
    class Program
    {
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "wikipedia-detox-250-line-data.tsv");
        static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "wikipedia-detox-250-line-test.tsv");
        static readonly string _modelpath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");

        public static async Task Main(string[] args)
        {
            var model = await Train();

            Evaluate(model);

            Console.ReadLine();
        }

        public static async Task<PredictionModel<SentimentData, SentimentPrediction>> Train()
        {
            var pipeline = new LearningPipeline();

            pipeline.Add(new TextLoader(_dataPath).CreateFrom<SentimentData>());
            pipeline.Add(new TextFeaturizer("Features", "SentimentText"));
            pipeline.Add(new FastTreeBinaryClassifier() { NumLeaves = 5, NumTrees = 5, MinDocumentsInLeafs = 2 });

            PredictionModel<SentimentData, SentimentPrediction> model = 
                pipeline.Train<SentimentData, SentimentPrediction>();

            await model.WriteAsync(_modelpath);

            return model;
        }

        public static void Evaluate(PredictionModel<SentimentData, SentimentPrediction> model)
        {
            var testData = new TextLoader(_testDataPath).CreateFrom<SentimentData>();
            var evaluator = new BinaryClassificationEvaluator();

            BinaryClassificationMetrics metrics = evaluator.Evaluate(model, testData);

            Console.WriteLine();
            Console.WriteLine("PredictionModel quality metrics evaluation");
            Console.WriteLine("------------------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.Auc:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
        }
    }
}
