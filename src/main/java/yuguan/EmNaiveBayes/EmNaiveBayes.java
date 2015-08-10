package yuguan.EmNaiveBayes;

import java.io.IOException;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map.Entry;
import java.util.Set;

import com.aliasi.classify.Classification;
import com.aliasi.classify.Classified;
import com.aliasi.classify.JointClassification;
import com.aliasi.classify.TradNaiveBayesClassifier;
import com.aliasi.corpus.Corpus;
import com.aliasi.corpus.ObjectHandler;
import com.aliasi.tokenizer.RegExTokenizerFactory;
import com.aliasi.util.Factory;

/**
 * 
 * @author zhouyang
 *
 */
class LabeledCorpus extends Corpus<ObjectHandler<Classified<CharSequence>>> {
	
	private List<String> X = new ArrayList<String>();
	private List<String> y = new ArrayList<String>();
	private final int N;
	
	public LabeledCorpus(List<String> X, List<String> y) {
		this.X = X;
		this.y = y;
		N = X.size();
	}

	@Override
	public void visitTrain(ObjectHandler<Classified<CharSequence>> classifier) {
		for (int i = 0; i < N; i++) {
			Classification label = new Classification(y.get(i));
			Classified<CharSequence> labeledInstance = 
					new Classified<CharSequence>(X.get(i), label);
			classifier.handle(labeledInstance);
		}
	}
	
	@Override
	public void visitTest(ObjectHandler<Classified<CharSequence>> handler) {
		throw new UnsupportedOperationException();
	}
}

/**
 * 
 * @author zhouyang
 *
 */
class UnlabeledCorpus extends Corpus<ObjectHandler<CharSequence>> {
	
	private List<String> X;
	
	public UnlabeledCorpus(List<String> X) {
		this.X = X;
	}
	
	@Override
	public void visitTrain(ObjectHandler<CharSequence> handler) {
		for (int i = 0; i < X.size(); i++)
			handler.handle(X.get(i));
	}
	
}

/**
 * 
 * @author zhouyang
 *
 */
public class EmNaiveBayes {

	/**
	 * Naive Baysian classifier
	 */
	private TradNaiveBayesClassifier mClassifier;
	/**
	 * Pseudo-count of Dirichlet prior for categories
	 */
	private final double CATEGORY_PRIOR;
	/**
	 * Pseudo-count of Dirichlet prior for words
	 */
	private final double TOKEN_PRIOR;
	/**
	 * To compensate for the effect of length. 
	 * In effect, if length(feature) > N, this pulls the estimate p(c|ws)
	 * closer to the category estimate p(c). If length(feature) < N, 
	 * it has the opposite effect, and actually increases the 
	 * attenuation.
	 */
	private final double LENGTH_NORM;
	/**
	 * Using text features that exceeding this count
	 */
	private final double MIN_TOKEN_COUNT;
	/**
	 * Minimum EM Improvment for continuing iteration
	 */
	private final double MIN_IMPROVMENT;
	/**
	 * Maximum iteration times
	 */
	private final int MAX_ITER;
	
	/**
	 * Construction EM Naive Bayes using default parameters
	 */
	public EmNaiveBayes() {
		this(1d, 1d, Double.NaN, 1, 1, 1000);
	}
	
	/**
	 * Construction EM Naive Bayes by assigning your parameters
	 * @param categoryPrior Pseudo-count of Dirichlet prior for categories
	 * @param tokenPrior Pseudo-count of Dirichlet prior for words
	 * @param lengthNorm To compensate for the effect of length
	 * @param minTokenCount Using text features that exceeding this count
	 * @param minImprovment Minimum EM Improvment for continuing iteration
	 * @param maxIterTimes Maximum iteration times
	 */
	public EmNaiveBayes(double categoryPrior, double tokenPrior, 
			double lengthNorm, double minTokenCount, double minImprovment,
			int maxIterTimes) {
		CATEGORY_PRIOR = categoryPrior;
		TOKEN_PRIOR = tokenPrior;
		LENGTH_NORM = lengthNorm;
		MIN_TOKEN_COUNT = minTokenCount;
		MIN_IMPROVMENT = minImprovment;
		MAX_ITER = maxIterTimes;
	}
	
	/**
	 * Training or fitting the model
	 * @param X List of text features separated by white space
	 * @param y List of labels, length must be the same with X's
	 */
	public void fit(List<String> X, List<String> y) {
		if (X.size() != y.size()) {
			throw new IllegalArgumentException("Dimension of X and y must"
					+ "be the same");
		}
		List<String> labeledX = new ArrayList<String>();
		List<String> unlabeledX = new ArrayList<String>();
		List<String> labeledy = new ArrayList<String>();
		
		for (int i = 0; i < X.size(); i++) {
			String y_i = y.get(i);
			if (y_i.compareTo("-1") == 0) {
				unlabeledX.add(X.get(i));
			} else {
				labeledX.add(X.get(i));
				labeledy.add(y_i);
			}
		}
		LabeledCorpus labeledCorpus = new LabeledCorpus(labeledX, labeledy);
		UnlabeledCorpus unlabeledCorpus = new UnlabeledCorpus(unlabeledX);
		
		// Initialize base classifier
		final Set<String> CATEGORIES = new HashSet<String>(y);
		final RegExTokenizerFactory TOKENIZER = new RegExTokenizerFactory("\\P{Z}+");
		TradNaiveBayesClassifier base = new TradNaiveBayesClassifier(
				CATEGORIES, TOKENIZER, CATEGORY_PRIOR, TOKEN_PRIOR,
				LENGTH_NORM);

		// Train initialize classifier
		labeledCorpus.visitTrain(base);
		
		// EM iteration
		Factory<TradNaiveBayesClassifier> factory = 
				new Factory<TradNaiveBayesClassifier>() {
			public TradNaiveBayesClassifier create() {
				TradNaiveBayesClassifier classifier = new TradNaiveBayesClassifier(
						CATEGORIES, TOKENIZER, CATEGORY_PRIOR, TOKEN_PRIOR, 
						LENGTH_NORM);
				return classifier;
			}
		};
		try {
			mClassifier = TradNaiveBayesClassifier.emTrain(
					base, 
					factory,  // new classifier factory while iteration
					labeledCorpus,
					unlabeledCorpus, 
					MIN_TOKEN_COUNT, 
					MAX_ITER,
					MIN_IMPROVMENT, 
					null  // reporter
					);
		} catch (IOException e) {
			System.err.print("System internal error!");
			System.exit(-1);
		} 
	}
	
	/**
	 * Predicting
	 * @param x white space separated text features
	 * @return Entry<String, Double>for (label, probability) pair
	 */
	public Entry<String, Double> predict(String x) {
		JointClassification jc = mClassifier.classify(x);
		String bestCategory = jc.bestCategory();
		double log2prob = jc.jointLog2Probability(1);
		return new AbstractMap.SimpleEntry<String, 
				Double>(bestCategory, log2prob);
	}
	
}
