package distances;

import writables.Point;

import java.util.Objects;

public abstract class Distance {
    abstract public double getDistance(Point p1, Point p2) throws Exception;
    abstract public Point getExpectation(Iterable<Point> points);

    public static Point sumPoints(Iterable<Point> points){
        Point result = null;
        for (Point point : points) {
            if (result == null) {
                result = new Point(point);
            } else {
                result.add(point);
            }
        }
        return result;
    }
    public static Distance getDistance(String name) {
        name = name.toLowerCase();
        if (Objects.equals(name, "euclidean")) {
            return new EuclideanDistance();
        } else if (Objects.equals(name, "cosine")) {
            return new CosineDistance();
        } else if (Objects.equals(name, "manhattan")) {
            return new ManhattanDistance();
        } else if (Objects.equals(name, "minkowski")) {
            return new MinkowskiDistance();
        }
        return null;
    }
}
