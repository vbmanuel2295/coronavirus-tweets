const natural = require("natural");
const csv = require("csvtojson");
const naiveBayes = natural.BayesClassifier;
const MODEL_PATH = "../001_train/coronavirus-tweet-classifier.json";

let loadFrozenModel = () => {
    return new Promise((resolve, reject) => {
        naiveBayes.load(MODEL_PATH, null, (err, classifier) => {
            if (err) {
                reject(err);
            }

            resolve(classifier);
        });
    });
};

let loadData = () => {
    return new Promise((resolve) => {
        csv()
            .fromFile("../001_train/dataset/Corona_NLP_test.csv")
            .then((json) => {
                resolve(json);
            });
    });
};

const transform = str => {
    return str.toLowerCase()
        .trim()
        .replace(/(?:https?|ftp):\/\/[\n\S]+/g, '')
        .replace(/[^a-zA-Z ]/g, "")
        .replace(/(\r\n|\n|\r)/gm, "");
}
(async () => {
    try {
        let [classifier, data] = await Promise.all([loadFrozenModel(), loadData()]);
        let results = data.map((i, idx) => {
            let predicted = classifier.classify(
                transform(i.OriginalTweet)
            );
            console.clear();
            console.log(`Inferring Item ${idx + 1} out of ${data.length}`);
            return {
                text: i.OriginalTweet,
                actual: i.Sentiment,
                predicted,
                isCorrect: i.Sentiment === predicted,
            };
        });
        let accurateHit = results.filter((i) => i.isCorrect);
        let missedItems = results.filter((i) => !i.isCorrect);
        let accuracy = (accurateHit.length / data.length) * 100;
        let errorRate = (missedItems.length / data.length) * 100;
        console.log(`Accurate: ${accurateHit.length} items`);
        console.log(`Missed: ${missedItems.length} items`);
        console.log(`Accuracy: ${accuracy.toFixed(2)}`);
        console.log(`Error Rate: ${errorRate.toFixed(2)}`);
    } catch (err) {
        console.log(err.message);
        console.log(err.stack);
    }
})();