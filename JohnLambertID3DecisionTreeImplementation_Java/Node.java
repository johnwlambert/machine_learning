import java.util.ArrayList;

public class Node {
		ArrayList<Node> children;
		Node Parent;
		double label; // really will be an integer, but the Matrix class stores all labels as type "double," so we say double here
		String attributeWeSplitOn;
		int attributesNumberThatWeSplitOn;
		//int[] histogramMostCommonFeatureValue;
		int[] mostCommonLabel;
		
		boolean isLeafNode; // use an extra variable that will help me prune, toggle this variable if I am making it the leaf temporarily
}
