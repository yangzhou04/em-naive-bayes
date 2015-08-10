package yuguan.EmNaiveBayes;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map.Entry;

import org.junit.Before;
import org.junit.Test;


public class EmNaiveBayesTest{

	private List<String> X = new ArrayList<String>();
	private List<String> y = new ArrayList<String>();
	
	@Before
	public void setUp() throws IOException {
		X.add("姚明 NBA 体育");
		X.add("姚明 金融 买卖");
		y.add("1");
		y.add("2");
		X.add("金融 招财宝");
		X.add("体育 CBA");
		y.add("-1");
		y.add("-1");
	}
	
	@Test
	public void testTrain() {
		EmNaiveBayes emNaiveBayes = new EmNaiveBayes();
		emNaiveBayes.fit(X, y);
		for (int i = 0; i < X.size(); i++) {
			Entry<String, Double> result = emNaiveBayes.predict(X.get(i));
			System.out.println(result.getKey() + ":" + y.get(i));
		}
	}
}
