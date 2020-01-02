import java.util.ArrayList;
import java.util.List;

public class CrossValidation {
    /*
     * Returns the k-fold cross validation score of classifier clf on training data.
     */
    public static double kFoldScore(Classifier clf, List<Instance> trainData, int k, int v) {
        // TODO : Implement

        // Accuracy : (number of correctly predicted instances / number of total instances)
        // k consecutive folds. In other words, the (0-indexed) i-th instance in a dataset of size n 
        // should belong to the ⌊i/(n/k)⌋-th fold.
        // double score = CrossValidation.kFoldScore(clf, trainData, k, vocabularySize(trainData));
        // v = vocabularySize(trainData)

        //1      [ 5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24]   [0 1 2 3 4]
        //2      [ 0  1  2  3  4 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24]   [5 6 7 8 9]
        //3      [ 0  1  2  3  4  5  6  7  8  9 15 16 17 18 19 20 21 22 23 24]   [10 11 12 13 14]
        //4      [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 20 21 22 23 24]   [15 16 17 18 19]
        //5      [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]   [20 21 22 23 24]
        
        // if k is 0 or 1, it is not k-fold
        if (k < 2) {
            return 0;
        }

        List<Instance> trainSet;
        List<Instance> testSet;
        ArrayList<Double> results = new ArrayList<>();

        // Test k times on k sets
        for (int i = 0; i < k; i++) {
            trainSet = CrossValidation.getTrainSet(trainData, k, i);
            testSet = CrossValidation.getTestSet(trainData, k, i);
            clf.train(trainSet, v);
            results.add(CrossValidation.validate(clf, testSet));
        }

        // Sum the score of each set
        double sumScores = 0;
        for (Double result : results) {
            sumScores += result;
        }

        // return the average score
        return sumScores / k;
    }


    /**
     * A Helper method to run the cross validation. Extract the test set.
     * @param trainData the whole data set
     * @param k the number of folds testing
     * @param fold the current fold being evaluated
     * @return the testing set
     */
    private static List<Instance> getTestSet(List<Instance> trainData, int k, int fold) {
        List<Instance> testSet = new ArrayList<>();

        // divide with 'k' to get the size
        // trainData is always divisible by k, so the size of each fold is same
        int setSize = trainData.size() / k;

        // index for extraction
        int index = setSize * fold;

        for (int i = index; i < (index + setSize); i++) {
            testSet.add(trainData.get(i));
        }

        return testSet;
    }

    /**
     * A Helper method. Extract the train set
     * @param trainData the whole data set
     * @param k the number of folds used
     * @param fold the current fold being evaluated
     * @return the training set
     */
    private static List<Instance> getTrainSet(List<Instance> trainData, int k, int fold) {
        List<Instance> trainSet = new ArrayList<>();

        // calculate the set size
        int setSize = trainData.size() / k;

        // Index for extraction: starting and ending point
        int head = setSize * fold;
        int tail = head + setSize;

        for (int i = 0; i < trainData.size(); i++) {
            if (!(i >= head && i < tail)) {
                trainSet.add(trainData.get(i));
            }
        }

        return trainSet;
    }

    /**
     * @param clf the classifier to test
     * @param testSet the test set to validate
     * @return the accuracy of the classifier
     */
    private static double validate(Classifier clf, List<Instance> testSet) {
        
        // If there is no set to test, score is 0
        if (testSet.size() == 0) {
            return 0;
        }

        // If the instance is correctly classified, we increment the count
        double numCorrect = 0;

        for (Instance inst : testSet) {
            ClassifyResult result = clf.classify(inst.words);
            if (result.label == inst.label) {
                numCorrect++;
            }
        }

        // Accuracy : (number of correctly predicted instances / number of total instances)
        return numCorrect / testSet.size();
    }

}
