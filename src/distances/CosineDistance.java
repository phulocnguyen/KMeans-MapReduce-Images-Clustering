package distances;

import writables.Point;

public class CosineDistance extends Distance {
    @Override
    public double getDistance(Point p1, Point p2) throws Exception {
        double[] p1Vector = p1.getVector();
        double[] p2Vector = p2.getVector();

        if (p1Vector.length != p2Vector.length) throw new Exception("Invalid length");

        double dotProduct = 0;
        double p1Norm = 0;
        double p2Norm = 0;
        
        for (int i = 0; i < p1Vector.length; i++) {
            dotProduct += p1Vector[i] * p2Vector[i];
            p1Norm += Math.pow(p1Vector[i], 2);
            p2Norm += Math.pow(p2Vector[i], 2);
        }

        // Cosine similarity = dotProduct / (norm(p1) * norm(p2))
        double cosineSimilarity = dotProduct / (Math.sqrt(p1Norm) * Math.sqrt(p2Norm));

        return 1 - cosineSimilarity;
    }

    @Override
    public Point getExpectation(Iterable<Point> points) {
        Point result = sumPoints(points);

        if (result != null) {
            result.compress();
        }
        return result;
    }
}
