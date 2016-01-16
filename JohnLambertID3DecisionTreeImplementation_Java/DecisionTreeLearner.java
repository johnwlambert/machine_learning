
/*
 * John W. Lambert
 * October 30, 2015
 * 
 * I implement an ID3 Decision Tree learning algorithm. I include the ability to handle
 * unknown attributes (You do not need to handle real valued attributes).  
 * I use standard information gain as my basic attribute evaluation metric.  
 * 
 * (Note that normal ID3 would always augment information gain with gain ratio 
 * or some other mechanism to penalize statistically insignificant attribute splits.  
 * Otherwise, even with approaches like pruning below, the SS# type of overfit could still hurt us.) 
 * 
 * For my own experiment, I decided to bin real-valued attributes into 5 bins.
 * I used the Iris data set and immediately made a new features matrix that
 * had 5 equally spaced bins, based on the calculated max and min values.
 * Then I can easily feed this through the decision tree learning algorithm.
 * (In theory, could also use it on the vowel data set.)
 * 
 * 	I implement my own tree using the Node.java class. We must use a recursive tree
 *  structure to store the decision tree.
 *  
 *  As I split my data set at particular nodes, the features and labels
 *  matrices still contain all of the old stuff that went down the other branches' paths.
 *  Our current branch, however, should not notice stuff that went down other branches.
 *  We solve this problem not by creating a lot of data sets, but rather by masking
 *  out the values we don't want to use. We mask out rows with "1s" in our skipRowsArray,
 *  and we mask out columns with "1s" in our skipColsArray. We make new versions of these
 *  arrays for every new branch we ever go down.
 */

import java.util.ArrayList;
import java.util.Random;
import java.lang.*;
import java.util.*;

public class DecisionTreeLearner extends SupervisedLearner {
	
	private int globalNodeCounter;
	
	Matrix binnedFeatureMatrix;
	Matrix globalOriginalFeatures;
	Matrix globalOriginalLabels;
	private Stack classStackOfNodesFromLeavesToRoot;
	private int[] classHistogramOfMostCommonValuesPerFeature;
	// root node has no parent
	private Node trainedDecisionTree;
	private double validationSetPercentageOfData = 0.80;
	
	public void train(Matrix features, Matrix labels) throws Exception {
		
		globalOriginalFeatures = features;
		globalOriginalLabels = labels;
		
//----------SPLIT OFF A VALIDATION SET-------------------------------------------------------
		// IMMEDIATELY SAVE SOME OF THIS, NEVER WILL TRAIN ON THESE, STICK THESE INTO A VALIDATION SET
		// ONCE accuracy starts to decrease ON THE VALIDATION SET, WE will stop the pruning
		int numRowsToGetIntoTrainingSet = (int)(features.rows() * validationSetPercentageOfData );
		Matrix featuresForTrainingTrimmed = new Matrix();
		featuresForTrainingTrimmed.setSize( numRowsToGetIntoTrainingSet , features.cols() );
		Matrix featuresValidationSet = new Matrix();
		featuresValidationSet.setSize( features.rows() - numRowsToGetIntoTrainingSet, features.cols() );
		Matrix labelsForTrainingTrimmed = new Matrix();
		labelsForTrainingTrimmed.setSize( numRowsToGetIntoTrainingSet , labels.cols() );
		Matrix labelsValidationSet = new Matrix();
		labelsValidationSet.setSize( features.rows() - numRowsToGetIntoTrainingSet, labels.cols() );
		// LOOP THROUGH AND PUT MOST OF FEATURES INTO featuresForTrainingTrimmed
		for( int row = 0; row < features.rows() ; row++ ) {
			for( int col = 0; col < features.cols() ; col++ ) {
				if( row < numRowsToGetIntoTrainingSet ) {
					featuresForTrainingTrimmed.set( row, col, features.get(row, col) );
				} else {
					featuresValidationSet.set( row - numRowsToGetIntoTrainingSet , col  , features.get(row, col) );
				}
			}
		}
		// LOOP THROUGH AND PUT MOST OF labels INTO labelsForTrainingTrimmed
		for( int row = 0; row < labels.rows() ; row++ ) {
			for( int col = 0; col < labels.cols() ; col++ ) {
				if( row < numRowsToGetIntoTrainingSet ) {
					labelsForTrainingTrimmed.set( row, col, labels.get(row, col) );
				} else {
					labelsValidationSet.set( row - numRowsToGetIntoTrainingSet , col  , labels.get(row, col) );
				}
			}
		}
		features = featuresForTrainingTrimmed;
		labels = labelsForTrainingTrimmed;
//-------------PERFORM THE TRAINING------------------------------------------------------------------------
		
		//features = createBinnedFeaturesMatrix( features );

		
		classHistogramOfMostCommonValuesPerFeature = new int[ features.cols() ];
		for( int col = 0; col < features.cols() ; col++ ) {
			classHistogramOfMostCommonValuesPerFeature[col] = (int)features.mostCommonValue(col); // STASH MOST COMMON VALUES
		}
		Node rootNode = new Node();
		int[] skipColsArray = new int[ features.cols() ];
		int[] skipRowsArray = new int[ features.rows() ];
		rootNode = makeTreeRec( rootNode, features, labels , skipColsArray, skipRowsArray);
		trainedDecisionTree = rootNode;
		double currentTrainingAccuracy = this.measureAccuracy( features, labels, null);
		System.out.println( "training set accuracy" +  currentTrainingAccuracy);
		System.out.println( "This many nodes before we prune" );
		countNodesInPrunedTree( trainedDecisionTree);
		createStackOfNodesFromBottomUp(); // pruning algorithm
		pruneTheDecisionTreeAndMeasureResults( featuresValidationSet, labelsValidationSet); 
		System.out.println("Tree is pruned");
		countNodesInPrunedTree( trainedDecisionTree);
		
		

	}
	
	/*
	 * propagate the data through the tree and predict on it
	 */
	public void predict(double[] features, double[] labels) throws Exception {
		globalNodeCounter = 1;
		Node childToExplore = new Node();
		if( trainedDecisionTree.children != null ) {
			double featureValueFromMatrix = features[ trainedDecisionTree.attributesNumberThatWeSplitOn ];
			if( featureValueFromMatrix == Double.MAX_VALUE) {
				featureValueFromMatrix = classHistogramOfMostCommonValuesPerFeature[ trainedDecisionTree.attributesNumberThatWeSplitOn ];
			}
			if( (int)featureValueFromMatrix < trainedDecisionTree.children.size() ) {
				childToExplore = trainedDecisionTree.children.get( (int)featureValueFromMatrix );
			} else {
				labels[0] = findArgMaxOfMostCommonValArray( trainedDecisionTree );
				//System.out.println( mostCommonWinner);
			}
			
		}
		// CHECK TO SEE IF isLeafNode is toggled on
		if( trainedDecisionTree.isLeafNode ) {
			labels[0] = findArgMaxOfMostCommonValArray( trainedDecisionTree);
		} else {
			double predictedLabel = exploreOneLevelInTree(features, childToExplore); // use recursion to look through array list
			labels[0] = predictedLabel;
			//System.out.println( predictedLabel);
		}
		//System.out.println(" this is depth " + globalNodeCounter );
	}
	
	// EASIER WAY TO DO THIS IS TO HAVE A SET OF INTEGERS, TEST MEMBERSHIP OF THE SET WITHOUT FOR LOOP, BUT WE WANTED AN INTEGER ARRAY

	/*
	 * This function goes deeper and deeper into the tree, each time following the branch that
	 * our features array corresponds to for each and every feature, until we reach a leaf
	 */
	private double exploreOneLevelInTree( double[] features, Node subTree ) {
		globalNodeCounter++;
		double predictedLabel = -1;
		if( subTree.children == null ) {  // base case
			return subTree.label;
		} else {
			Node childToExplore = new Node();
			if( subTree.children != null ) {
				double featureValueFromMatrix = features[ subTree.attributesNumberThatWeSplitOn ];
				if( featureValueFromMatrix == Double.MAX_VALUE) {
					featureValueFromMatrix = classHistogramOfMostCommonValuesPerFeature[ trainedDecisionTree.attributesNumberThatWeSplitOn ];
				}
				if( (int)featureValueFromMatrix < subTree.children.size() ) {
					childToExplore = subTree.children.get( (int)featureValueFromMatrix );
				} else {
					return findArgMaxOfMostCommonValArray( subTree);
				}				
			}
			
			// CHECK TO SEE IF isLeafNode is toggled on
			if( subTree.isLeafNode ) {
				 return findArgMaxOfMostCommonValArray( subTree);
			} else {
				predictedLabel = exploreOneLevelInTree(features, childToExplore); // use recursion to look through array list
			}
		}
		//System.out.println( "intermediate to pass up" + predictedLabel);
		return predictedLabel; 
	}
	
	private Node makeTreeRec( Node currentNode, Matrix features, Matrix labels , int[] skipColsArray, int[] skipRowsArray) {
		currentNode.children = new ArrayList<Node>();
		int featuresLeftToSplitOn = 0;
		
		currentNode.mostCommonLabel = new int[ globalOriginalLabels.valueCount(0) ];
		for( int feature = 0; feature < skipColsArray.length; feature++ ) {
			if( skipColsArray[feature] != 1) {
				featuresLeftToSplitOn++;
			}
		}
		Set<Double> labelValueCount = new HashSet<Double>();
		for( int row = 0; row < labels.rows(); row++ ) {
			if( skipRowsArray[row] != 1) {
				labelValueCount.add( labels.get(row, 0) );
				currentNode.mostCommonLabel[ (int)globalOriginalLabels.get(row,0 ) ]++;
			}
		}
		// check to see if there are no more instances to use for future splits, I ran out of data
		
		if( labelValueCount.size() == 1 ) {		// if( all classes have the same label )
			Node leaf = new Node();
			leaf.label = labelValueCount.iterator().next(); // return a leaf with that label // do i need to say that leaf.parent = currentNode
			return leaf;
		}
		else if( featuresLeftToSplitOn == 0 ) {   // there are no features left to test, BELOW IS THE BASELINE LEARNER
			
			int[] histogram = new int[ globalOriginalLabels.valueCount(0) ]; 	// make a histogram across all of the labels
			for( int row = 0; row < features.rows(); row++ ) {
				boolean rowNotApplicable = false;
				if( skipRowsArray[row] == 1 ) {
					rowNotApplicable = true;
				}
				if( rowNotApplicable == true ) { // continue if the value in the skipRowsArray was 1 here
					continue;
				}
				int labelAtThisRow = (int) labels.get( row , 0 ); // yes they are doubles, but they are actually integers
				histogram[labelAtThisRow]++;
			}	
			int majorityClass = -1;
			for( int i = 0; i < histogram.length; i++ ) {
				if( histogram[i] > majorityClass ) {
					majorityClass = histogram[i];
				}
			}
			Node majorityClassLeaf = new Node();
			majorityClassLeaf.label = majorityClass;
			return majorityClassLeaf; // return a leaf with the most common label
		} else {
			int attributeID = chooseFeatureToSplitOn ( features, labels, skipColsArray , skipRowsArray);
			
			if( attributeID == -9000 ) return currentNode; // we see we didn't meet gain threshold
			currentNode.attributeWeSplitOn = globalOriginalFeatures.m_attr_name.get(attributeID); // Do I do this for the CurrentNode or do I do it for the subTree node??? And how do I get an int out of there?
			currentNode.attributesNumberThatWeSplitOn = attributeID;
			skipColsArray[attributeID] = 1;
			int[] newSkipColsArray = new int[ skipColsArray.length ]; 	// make a new skipRowsArray for every single branch
			for( int featureValue = 0; featureValue < globalOriginalFeatures.valueCount(attributeID); featureValue++ ) { // add a branch from the node for each possible value f in that Feature That We just decided to split on
				for( int col = 0; col < newSkipColsArray.length; col++ ) {  
					newSkipColsArray[col] = skipColsArray[col];
				}
				Node subTree = new Node();
				int[] newSkipRowsArray = new int[ skipRowsArray.length ]; 	// make a new skipRowsArray for every single branch
				//int[] histogramMostCommonFeatureValue = new int[ features.valueCount(attributeID) ]; 
				int instanceLeftCounter = 0;
				for( int row = 0; row < newSkipRowsArray.length; row++ ) {  
					
					newSkipRowsArray[row] = skipRowsArray[row];
					if( features.get(row, attributeID) != featureValue) {
						newSkipRowsArray[row] = 1;	// calculate Sf by removing F from the set of features
					} 
					if( newSkipRowsArray[row] != 1) instanceLeftCounter++;
				}
				if( instanceLeftCounter == 0) {
					subTree.label = findArgMaxOfMostCommonValArray( currentNode);
					continue;
				}
				subTree = makeTreeRec( subTree , features, labels, newSkipColsArray, newSkipRowsArray ); // recursively call the algorithm with Sf, to compute the gain relative to the current set of examples
									// should I be passing in the next node here?
				currentNode.children.add( subTree );
			}
		}
		return currentNode;
	}
	
	/*
	 * We MAXIMIZE GAIN BY MINIMIZING featureInfo/ featureEntropy
	 * We take the attribute that will minimize the entropy of the feature we split on.
	 */
	private int chooseFeatureToSplitOn( Matrix features, Matrix labels , int[] skipColsArray, int[] skipRowsArray) {
		
		int attributeID = -1; // JUST ARBITRARY IMPOSSIBLE VALUE TO INITIALIZE
		double minFeatureEntropy = Double.MAX_VALUE;	// maximumEntropy
		
		for( int feature = 0; feature < features.cols(); feature++ ) {
			boolean treeAlreadyPartitionedOnThisFeature = false;
			if( skipColsArray[feature] == 1 ) {
				treeAlreadyPartitionedOnThisFeature = true;
			}
			if( treeAlreadyPartitionedOnThisFeature == true ) {
				continue;
			}
			double currentFeatureEntropy = findEntropyOfParticularAttribute(features, labels, feature , skipRowsArray); // do i really need to pass in attributeId
			if( currentFeatureEntropy < minFeatureEntropy ) {
				attributeID  = feature;
				minFeatureEntropy = currentFeatureEntropy;
			}
		}
		
		//System.out.println( "this is the min feature entropy" + minFeatureEntropy);
		//if( minFeatureEntropy > 0.3) return -9000;
		return attributeID;
	}
	
	
	
	/* 
	 * Info(S) = Entropy(S) = -Sigma Pi * log2 ( Pi)
	 * 
	 *  	           A
	 * 		     /     |    \
	 * 	        /      |      \
	 * 	      /        |        \
	 *      A= cat   A=dog    A = horse
	 * Class entropy   Class entropy   Class entropy
	 * 
	 * Other option to store the HISTOGRAM is a Map, HashMap<Int,Int> histogram = new HashMap<Int, Int>(); 
	 * TURN THE HISTOGRAM INTO A PROBABILITY DISTRIBUTION
	 * Map<Integer, Double> histogram = new TreeMap<Int, Double>();
	 * double[] histogram = new double[ labels.valueCount(0) ];
 	 * But I chose the array option
	 */
	private double findEntropyOfParticularAttribute( Matrix features, Matrix labels, int attributeID, int[] skipRowsArray) {
		
		double score = 0;
		int totalNumInstancesForThisAttribute = 0;  // we could just set this equal to the number of rows that are not supposed to be skipped
		for( int specificFeatureValue = 0; specificFeatureValue < globalOriginalFeatures.valueCount(attributeID); specificFeatureValue++ ) {
			double entropyForSpecificFeatureValue = 0;
			double[] histogram = new double[ globalOriginalLabels.valueCount(0) ]; // just increment the bins each time, much easier
			int numOfInstancesForThisFeatureValue = 0; // THIS IS TOTAL AMOUNT OF STUFF INSIDE OF THE HISTOGRAM
			
			// iterate over feature matrix, see which ones have this value, how many of them are there?
			for( int row = 0; row < features.rows(); row++ ) {
				boolean rowNotApplicable = false;
				if( skipRowsArray[row] == 1 ) {
					rowNotApplicable = true;  	// SHOULDN'T I CONTINUE IF THE VALUE IN THE SKIP ROWS ARRAY WAS 1 HERE?
				}
				if( rowNotApplicable == true ) {
					continue;
				}
				if( features.get(row, attributeID) == specificFeatureValue ) {
					numOfInstancesForThisFeatureValue++;
					int labelAtThisRow = (int) labels.get( row , 0 ); // yes they are doubles, but they are actually integers
						histogram[labelAtThisRow]++;
				}
			}		
			totalNumInstancesForThisAttribute += numOfInstancesForThisFeatureValue;
			for( int histogramIndex = 0; histogramIndex < histogram.length; histogramIndex++ ) {
				if( numOfInstancesForThisFeatureValue != 0 ) { // check if we will divide by 0 in the next line
					histogram[histogramIndex] /= numOfInstancesForThisFeatureValue ; 			//	The histogram now represents a probability distribution that sums up to 1
				}
				if( histogram[histogramIndex] != 0) {
					entropyForSpecificFeatureValue += ( histogram[histogramIndex] * (Math.log( histogram[histogramIndex] ) / Math.log(2) ) );
				}
			}
			score += ( entropyForSpecificFeatureValue * numOfInstancesForThisFeatureValue);	
		}
		score /= ( -1 * totalNumInstancesForThisAttribute );
		return score; // which is the entropy if we split on this attribute;	
	}
	

	
	/*
	 * We put all of the nodes, leaves first, then working our way down to the root, onto a stack.  
	 * We manhandle the recursion with our stack and queue data structures, which is amazing.
	 * 
	 * The stack now contains at its top, the very last leaf of the tree.
	 * Most deeply embedded in the stack is the root node. We want the nodes
	 * ordered in this manner so that we can prune from the bottom up and 
	 * see incremental changes in our decision tree as we seek to avoid overfit.
	 */
	void createStackOfNodesFromBottomUp() {
		Stack stackOfNodesFromLeavesToRoot = new Stack(); //initialize stack
		// we have a root node
		Queue<Node> myQueue = new LinkedList<Node>();// initialize a queue
		myQueue.add( trainedDecisionTree ); // add the root node to the queue
		while ( myQueue.size() > 0 ) { // while (!myQueue.isEmpty())
			Node n = myQueue.remove(); // takes the first element off of the queue, pop
			ArrayList<Node> childrenToAdd = n.children;
			if( childrenToAdd != null) {
				myQueue.addAll( childrenToAdd ); //n.children );
			}
			stackOfNodesFromLeavesToRoot.push( n );
		}
		classStackOfNodesFromLeavesToRoot = stackOfNodesFromLeavesToRoot;
	}
	
	
	/*
	 * We pass in the validation set to this function
	 * Prune the leaves by calling measureAccuracy
	 * My node class contains a field: Node.isLeafNode
	 * I use this extra variable that will help me prune, toggle this variable if I am making it the leaf temporarily
	 * 
	 * Validation stopping could stop too early (e.g. higher order combination)
	Pruning a full tree (one where all possible nodes have been added)
	Prune any nodes which would not hurt accuracy
	Could allow some higher order combinations that would have been missed with validation set early stopping (though could do a VS window)
	Can simultaneously consider all nodes for pruning rather than just the current frontier
	Train tree out fully (empty or consistent partitions or no more attributes)
	For EACH non-leaf node, test accuracy on a validation set for a modified tree where the sub-tree of the node is removed and the node is assigned the majority class based on the instances it represents from the training set
	Keep pruned tree which does best on the validation set and does at least as well as the original tree on the validation set
	Repeat until no pruned tree does as well as the current tree
	
	call measureAccuracy on the class, which will call predict using the newly updated decision tree
	did the accuracy on the validation set get worse?

	 */
	void pruneTheDecisionTreeAndMeasureResults( Matrix featuresValidationSet, Matrix labelsValidationSet ) throws Exception {
		double baselineValidationSetAccuracy = this.measureAccuracy( featuresValidationSet,labelsValidationSet, null );
		System.out.println( "THIS IS BASELINE VALID SET ACCURACY " + baselineValidationSetAccuracy );
		while( !classStackOfNodesFromLeavesToRoot.isEmpty() ) {
			Node nodeToBeConvertedToLeaf = (Node)classStackOfNodesFromLeavesToRoot.pop(); // do I have to cast it?
			//nodeToBeConvertedToLeaf.isLeafNode = true; 		// toggle variable at one position, call if leaf
			if( this.measureAccuracy( featuresValidationSet,labelsValidationSet, null ) < baselineValidationSetAccuracy ) { //will run on whole decision tree
				nodeToBeConvertedToLeaf.isLeafNode = false;// our accuracy is now decreasing, dont prune here
			} else {
				double newBetterAccuracy = this.measureAccuracy( featuresValidationSet,labelsValidationSet, null ); // replace with higher value
				//System.out.println( "pruning helped" + " new accuracy " + newBetterAccuracy );
				baselineValidationSetAccuracy = newBetterAccuracy;
			}
			 // toggle variable back to the way it was before, not being a leaf anymore
		}
	}
	
	
	/*
	 * Allow for continuously-valued feature values (We can now run on iris and vowel data sets).
	 * 
	 * We store the binned features matrix inside a private instance variable of the class
	 * We will draw from it, but then always draw from features matrix when we want attribute names, etc
	 */
	
	
	/*
	 * python argmax
	 */
	 int findArgMaxOfMostCommonValArray( Node subTree) {
		 int maxFrequency = 0;
		int mostCommonWinner = -1;
		for(int i = 0; i < subTree.mostCommonLabel.length ; i++ ) {
			if( subTree.mostCommonLabel[i] > maxFrequency ) {
				maxFrequency = subTree.mostCommonLabel[i];
				mostCommonWinner = i;
			}
		}
		return mostCommonWinner;
	 }
	 
	 void countNodesInPrunedTree(Node decisionTree) {
		 	Stack stackOfNodesFromLeavesToRoot = new Stack(); //initialize stack
			// we have a root node
			Queue<Node> myQueue = new LinkedList<Node>();// initialize a queue
			myQueue.add( decisionTree ); // add the root node to the queue
			while ( myQueue.size() > 0 ) { // while (!myQueue.isEmpty())
				Node n = myQueue.remove(); // takes the first element off of the queue, pop
				ArrayList<Node> childrenToAdd = n.children;
				if( (childrenToAdd != null) && (n.isLeafNode==false) ) {
					myQueue.addAll( childrenToAdd ); //n.children );
				}
				stackOfNodesFromLeavesToRoot.push( n );
			}
			System.out.println( "This many nodes in the pruned tree" + stackOfNodesFromLeavesToRoot.size() );
	 }
}
