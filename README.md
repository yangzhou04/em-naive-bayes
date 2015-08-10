# Semi-supervised text classification using EM NaiveBayesian Classifier

This is an Java Implementation of Semi-supervised text classification using EM NaiveBayesian Classifier 
based on [Lingpipe](http://alias-i.com/lingpipe/)

Currently, there is no command line style of envoking. Just clone the code and try:

```java
// Create labelling and unlabelling corpus
List<String> X = new ArrayList<String>();
List<String> y = new ArrayList<String>();
X.add("YaoMing NBA");  y.add("1"); // add text features and labels
X.add("Yaoming Selling");  y.add("2");
X.add("Selling Zhaocaibao"); y.add("-1"); // add unlabeled text features, "-1" is marked as unlabeled
X.add("NBA CBA"); y.add("-1"); // length of X and y must be the same

EmNaiveBayes emNaiveBayes = new EmNaiveBayes();
// training
emNaiveBayes.fit(X, y);
// pridiction
for (int i = 0; i < X.size(); i++) {
	Entry<String, Double> result = emNaiveBayes.predict(X.get(i));
	System.out.println(result.getKey() + ":" + y.get(i));
}
```
