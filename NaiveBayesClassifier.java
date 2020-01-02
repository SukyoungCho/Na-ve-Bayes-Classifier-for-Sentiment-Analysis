import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Your implementation of a naive bayes classifier. Please implement all four methods.
 */

public class NaiveBayesClassifier implements Classifier {

	private Map<Label, Integer> docsPerLabel;
    private Map<Label, Integer> wordsPerLabel;
    private Map<String, Integer> posWords;
    private Map<String, Integer> negWords;
    private double totalDocs;
    private double vocabSize;

    /**
     * Trains the classifier with the provided training data and vocabulary size
     */
    @Override
    public void train(List<Instance> trainData, int v) {
        // TODO : Implement
        // Hint: First, calculate the documents and words counts per label and store them. 
        // Then, for all the words in the documents of each label, count the number of occurrences of each word.
        // Save these information as you will need them to calculate the log probabilities later.
        //
        // e.g.
        // Assume m_map is the map that stores the occurrences per word for positive documents
        // m_map.get("catch") should return the number of "catch" es, in the documents labeled positive
        // m_map.get("asdasd") would return null, when the word has not appeared before.
        // Use m_map.put(word,1) to put the first count in.
        // Use m_map.replace(word, count+1) to update the value
        this.wordsPerLabel = this.getWordsCountPerLabel(trainData);
        this.docsPerLabel = this.getDocumentsCountPerLabel(trainData);
        this.posWords = this.getWordCountsPerLabel(Label.POSITIVE, trainData);
        this.negWords = this.getWordCountsPerLabel(Label.NEGATIVE, trainData);
        this.totalDocs = trainData.size();
        this.vocabSize = v;
    }

    /*
     * Counts the number of words for each label
     */
    @Override
    public Map<Label, Integer> getWordsCountPerLabel(List<Instance> trainData) {
        // TODO : Implement
        Map<Label, Integer> words = new HashMap<>();
        int pos = 0;
        int neg = 0;

        for (Instance inst : trainData) {
            if (inst.label == Label.POSITIVE) {
                pos += inst.words.size();
            }
            else {
                neg += inst.words.size();
            }
        }

        words.put(Label.POSITIVE, pos);
        words.put(Label.NEGATIVE, neg);

        return words;
    }


    /*
     * Counts the total number of documents for each label
     */
    @Override
    public Map<Label, Integer> getDocumentsCountPerLabel(List<Instance> trainData) {
        // TODO : Implement
        Map<Label, Integer> docs = new HashMap<>();
        Integer pos = 0;
        Integer neg = 0;

        for (Instance inst : trainData) {
            if (inst.label == Label.POSITIVE) {
                pos++;
            }
            else {
                neg++;
            }
        }

        docs.put(Label.POSITIVE, pos);
        docs.put(Label.NEGATIVE, neg);
        
        return docs;
    }

    /**
     * This is a helper method to count the number of words for the given label such as positive
     * or negative
     * @param label the label to count the number of
     * @param trainData the training data
     * @return a map with the label as keys and counts as values
     */
    private Map<String, Integer> getWordCountsPerLabel(Label label, List<Instance> trainData) {
        Map<String, Integer> wordCount = new HashMap<>();

        // Loop through the datast and check if the word accords with the given label
        for (Instance inst : trainData) {
            if (inst.label == label) {
                List<String> words = inst.words;
                for (String word : words) {
                    // Increment the count the word is alrady in the lise,
                    // Else, add to the list
                    if (wordCount.containsKey(word)) {
                        int count = wordCount.get(word);
                        // increase the count
                        wordCount.replace(word, count + 1);
                    } else {
                    	// add to the list
                        wordCount.put(word, 1);
                    }
                }
            }
        }

        return wordCount;
    }


    /**
     * Returns the prior probability of the label parameter, i.e. P(POSITIVE) or P(NEGATIVE)
     */
    private double p_l(Label label) {
        // TODO : Implement
        // Calculate the probability for the label. No smoothing here.
        
        // zero probability if the docs is empty
        if (this.totalDocs == 0) {
            return 0;
        }

        double numLabel = this.docsPerLabel.get(label);
        
        // Just the number of label counts divided by the number of documents.
        return numLabel / this.totalDocs;
    }

    /**
     * Returns the smoothed conditional probability of the word given the label, i.e. P(word|POSITIVE) or
     * P(word|NEGATIVE)
     */
    private double p_w_given_l(String word, Label label) {
        // TODO : Implement
        // Calculate the probability with Laplace smoothing for word in class(label)
        
        double wordCount;
        double tokens = this.wordsPerLabel.get(label);

        // Get the numer of tokens and words with the given label
        if (label == Label.NEGATIVE) {
            wordCount = this.negWords.getOrDefault(word, 0);
        } else {
            wordCount = this.posWords.getOrDefault(word, 0);
        }

        // Laplace Smoothing
        double delta = 1;
        double numerator = wordCount + delta;
        double denominator = (this.vocabSize * delta) + tokens;

        // To avoid the infinity
        if (denominator == 0.0) {
            return 0.0;
        }

        return numerator / denominator;
    }

    /**
     * Classifies an array of words as either POSITIVE or NEGATIVE.
     */
    @Override
    public ClassifyResult classify(List<String> words) {
        // TODO : Implement
        // Sum up the log probabilities for each word in the input data, and the probability of the label
        // Set the label to the class with larger log probability

        // Result
        ClassifyResult result = new ClassifyResult();
        
        // create a map for pairs of probabiltiy and given label
        Map<Label, Double> probMap = new HashMap<>();

        // sum of conditional probabilities
        double sumCondNeg = 0;
        double sumCondPos = 0;

        // Calculate the positive conditional probability
        double probPos = this.p_l(Label.POSITIVE) == 0.0 ? 0.0
                : Math.log(this.p_l(Label.POSITIVE));
        for (String word: words) {
            sumCondPos += Math.log(this.p_w_given_l(word, Label.POSITIVE));
        }
        probMap.put(Label.POSITIVE, probPos + sumCondPos);

        // the negative conditional probability
        double probNeg = this.p_l(Label.NEGATIVE) == 0.0 ? 0.0
                : Math.log(this.p_l(Label.NEGATIVE));
        for (String word : words) {
            sumCondNeg += Math.log(this.p_w_given_l(word, Label.NEGATIVE));
        }
        probMap.put(Label.NEGATIVE, probNeg + sumCondNeg);
        
        // Classification
        if (probMap.get(Label.POSITIVE) > probMap.get(Label.NEGATIVE)) {
            result.label = Label.POSITIVE;
        } else {
        	result.label = Label.NEGATIVE;
        }

        result.logProbPerLabel = probMap;

        return result;
    }


}
