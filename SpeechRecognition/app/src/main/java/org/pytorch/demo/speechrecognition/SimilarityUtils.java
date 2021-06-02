package org.pytorch.demo.speechrecognition;

import java.util.Arrays;

public class SimilarityUtils {

    public static void inplaceExp(double[] a, int start, int end) {
        assert (start >= 0);
        assert (end >= start);
        assert (end <= a.length);
        double sum = 0.0;
        for (int i = start; i < end; i++) {
            a[i] = Math.exp(a[i]);
        }
    }

    public static void inplaceLogSoftmax(double[] a, int start, int end) {
        assert (start >= 0);
        assert (end >= start);
        assert (end <= a.length);
        double sum = 0.0;
        double max = Arrays.stream(a, start, end).max().getAsDouble();
        for (int i = start; i < end; i++) {
            a[i] = Math.exp(a[i]-max);
            sum += Math.exp(a[i]-max);
        }
        double logsumexp = Math.log(sum) + max;
        for (int i = start; i < end; i++) {
            a[i] -= logsumexp;
        }
    }

    public static void inplaceSoftmax(double[] a, int start, int end) {
        assert (start >= 0);
        assert (end >= start);
        assert (end <= a.length);
        double sum = 0.0;
        for (int i = start; i < end; i++) {
            a[i] = Math.exp(a[i]);
            sum += a[i];
        }
        for (int i = start; i < end; i++) {
            a[i] /= sum;
        }
    }

    public static double dotProduct(double[] a, int startA, double[] b, int startB, int len) {
        double sum = 0;
        for (int i=0;i<len;i++) {
            sum += a[startA+i] * b[startB+i];
        }
        return sum;
    }

    public static double[] copyRegion(double[] a, int start, int end) {
        return Arrays.copyOfRange(a, start, end);
    }

    public static double[] copyRegion(float[] a, int start, int end) {
        double[] result = new double[end-start];
        for (int i=start;i<end;i++) {
            result[i-start] = a[i];
        }
        return result;
    }

    public static double cosineSimilarity(double[] a, int startA, double[] b, int startB, int len) {
        return dotProduct(a, startA, b, startB, len) / ( Math.sqrt(dotProduct(a,startA, a, startA, len)) * Math.sqrt(dotProduct(b,startB, b, startB, len)));
    }

    public static double KLDivergenceLog(double[] a, int startA, double[] b, int startB, int len) {
        // Assuming both distributions are already normalized via LogSoftmax
        //  sum ( exp(a[i] * ( a[i] - b[i] ) ) )
        double tmp = 0.0;
        for (int i=0;i<len;i++) {
            if (Double.isFinite(a[i+startA])) {
                double weight = Math.exp(a[i+startA]);
                if (weight>0.0) {
                    tmp += weight * ( a[i+startA] - b[i+startB] );
                }
            }
        }
        return tmp;
    }

    public static double KLDivergence(double[] a, int startA, double[] b, int startB, int len) {
        // Assuming both distributions are already normalized via Softmax
        // sum ( a[i] * ( log(a[i]) - log(b[i]) ) ) )
        double tmp = 0.0;
        for (int i=0;i<len;i++) {
            if (a[i+startA]>0.0) {
                double weight = a[i+startA];
                if (weight>0.0) {
                    tmp += weight * ( Math.log(a[i+startA]) - Math.log(b[i+startB]) );
                }
            }
        }
        return Math.log(tmp);
    }

    public static double JensenShannonDivergenceLog(double[] a, int startA, double[] b, int startB, int len) {
        return (KLDivergenceLog(a, startA, b, startB, len) + KLDivergenceLog(b, startB, a, startA, len)) / (2.0*Math.log(2.0));
    }

    public static double JensenShannonDivergence(double[] a, int startA, double[] b, int startB, int len) {
        return (KLDivergence(a, startA, b, startB, len) + KLDivergence(b, startB, a, startA, len)) / (2.0*Math.log(2.0));
    }


}
