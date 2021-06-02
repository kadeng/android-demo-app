package org.pytorch.demo.speechrecognition;

import smile.math.distance.Distance;
import smile.math.distance.DynamicTimeWarping;
import smile.math.distance.EuclideanDistance;
import smile.math.distance.JensenShannonDistance;
import smile.math.distance.Metric;

public class Word2VecRecognition {
    public static double MAX_WARP_PERCENTAGE = 100.0;

    public String[] alphabet;
    public double[][] recognitionMatrix;
    public final int tokenCount;
    public final int alphabetSize;
    private Metric<double[]> tokenDistance;
    private Distance<double[][]> recognitionDistance;
    public String name;

    public Word2VecRecognition(final float values[], final String[] alphabet, String name) {
        assert(values.length % alphabet.length==0);
        this.alphabet = alphabet;
        this.alphabetSize = alphabet.length;
        this.tokenCount = values.length / alphabet.length;
        this.recognitionMatrix = new double[tokenCount][];
        for (int i=0;i<tokenCount;i++) {
            recognitionMatrix[i] = SimilarityUtils.copyRegion(values, i*alphabetSize, (i+1)*alphabetSize);
            SimilarityUtils.inplaceSoftmax(recognitionMatrix[i], 0, alphabetSize);
        }
        this.tokenDistance = new JensenShannonDistance();
        this.recognitionDistance = new DynamicTimeWarping<>(this.tokenDistance, MAX_WARP_PERCENTAGE /100.0);
        this.name = name;
    }

    public double distance(Word2VecRecognition y) {
        return this.recognitionDistance.d(this.recognitionMatrix, y.recognitionMatrix);
    }
}
