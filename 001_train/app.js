/**
 * This module is used for training a coronavirus tweets classifier.
 *
 * 1. It loads data from the dataset/sample.csv
 * 2. It Translates the numeric class IDs to string based ones.
 * 3. It freezes the model's weights inside the ./coronavirus-tweet-classifier.json
 */
const natural = require("natural");
const csv = require("csvtojson");

let persistModel = (classifier) => {
    return new Promise((resolve, reject) => {
        classifier.save("./coronavirus-tweet-classifier.json", (err) => {
            if (err) {
                reject(err);
            } else {
                resolve();
            }
        });
    });
};

let loadData = () => {
    return new Promise((resolve) => {
        csv()
            .fromFile("./dataset/Corona_NLP_train.csv")
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
        let dataset = await loadData();
        let classifier = new natural.BayesClassifier();
        dataset.forEach((i) => {
            classifier.addDocument(
                transform(i.OriginalTweet),
                i.Sentiment
            );
        });
        classifier.train();
        await persistModel(classifier);
    } catch (err) {
        console.log(err.message);
        console.log(err.stack);
    }
})();