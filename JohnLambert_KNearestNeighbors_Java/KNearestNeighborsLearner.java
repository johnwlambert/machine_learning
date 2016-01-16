

/*
 * John Lambert
 * November 13, 2015
 * 
 * I implement K-Nearest Neighbors.
 * 
 * I implement the k nearest neighbor algorithm and the k nearest neighbor regression algorithm,
 * including optional distance weighting for both algorithms.  I use a Euclidean distance
 * metric.
 * 
 * I also implement the leave-one-out reduction algorithm to reduce the number of instances
 * I predict on to about 25% of original data set size.
 */

import java.util.ArrayList;
import java.util.Random;
import java.lang.*;
import java.util.*;
	
public class KNearestNeighborsLearner extends SupervisedLearner {

	public Matrix savedFeaturesMatrix;
	public Matrix savedLabelsMatrix;
	public int kValue = 13; // start at 1, 3, 5, 7, 9, 11, 13, etc.
	public boolean useRegression = false;
	public boolean useWeightedDistance = true;
	public int[] globalMaskedOutFeatures;
	public int numInstancesUsed;

	
	/*
	 * Save the features and the labels.
	 * 
	 * If my algorithm can predict the point correctly (When it is not included), 
	 * then we mask this instance out. This works for classification.
	 */
	public void train(Matrix features, Matrix labels) throws Exception {
		savedFeaturesMatrix = features;
		savedLabelsMatrix = labels;
		
// --------BELOW WE EXECUTE LEAVE-ONE-OUT-REDUCTION TECHNIQUES----------------------------
		globalMaskedOutFeatures = new int[ features.rows() ];
		for( int i = 0; i < globalMaskedOutFeatures.length; i++ ) {
			globalMaskedOutFeatures[i] = 0;
		}
		numInstancesUsed = features.rows();
		
		for( int trainingInstance = 0; trainingInstance < savedFeaturesMatrix.rows(); trainingInstance++ ) {
			globalMaskedOutFeatures[trainingInstance] = 1; // act like this instance doesn't exist
			double[] featuresToCheckIfCanMaskOut = new double[ savedFeaturesMatrix.cols() ];
			for( int col = 0; col < savedFeaturesMatrix.cols(); col++ ) {
				featuresToCheckIfCanMaskOut[col] = savedFeaturesMatrix.get( trainingInstance, col);
			}
			ArrayList<KNNPoint> distanceTable = new ArrayList<KNNPoint>();
			distanceTable = initializeDistanceTableWithZeroValues( distanceTable );
			distanceTable = makeDistanceLookupTable( distanceTable, featuresToCheckIfCanMaskOut );
			distanceTable = sortTheDistanceLookupTable( distanceTable);
			double predictedResult = extractLabelWithMostVotes( distanceTable );
			if( predictedResult == savedLabelsMatrix.get(trainingInstance, 0) ) {
				numInstancesUsed--;	// we can predict the class correctly even without including this instance
			} else {
				globalMaskedOutFeatures[trainingInstance] = 0; // turns out this instance was necessary
			}
		}
		System.out.println( "We were able to process all of the masking on the features");
		System.out.println( "We use " + numInstancesUsed + " out of a total of " + features.rows() + " instances.");
// ------------ END OF LEAVE-ONE-OUT-REDUCTION------------------------------
	}
	
	
	/*
	 * We make a lookup table for every single instance, with calculated distances away from it
	 * We then sort that lookup table
	 * We then take the k top values from that lookup table
	 * Put their votes into bins, and see which bin has most votes in it
	 * Return that bin's label
	 */
	public void predict(double[] features, double[] labels) throws Exception {
		
		ArrayList<KNNPoint> distanceTable = new ArrayList<KNNPoint>();
		distanceTable = initializeDistanceTableWithZeroValues( distanceTable );
		distanceTable = makeDistanceLookupTable( distanceTable, features );
		distanceTable = sortTheDistanceLookupTable( distanceTable);
		labels[0] = extractLabelWithMostVotes( distanceTable );
	}
	
	
	/*
	 * One algorithm for weighted distance between two points
	 * Another algorithm for unweighted distanced between two points
	 * BECAUSE square root of (X) is monotonically increasing function,
	 * never actually compute the square root, just store the square, then we'll sort that
	 */
	private ArrayList<KNNPoint> makeDistanceLookupTable( ArrayList<KNNPoint> distanceTable, double[] featuresForPrediction ) {
		int counterOfRowWeAreInsertingAt = 0;
		for( int trainingInstance = 0; trainingInstance < savedFeaturesMatrix.rows(); trainingInstance++ ) {
			if( globalMaskedOutFeatures[trainingInstance] == 1 ) {
				continue; // leave one out reduction, this was already predicted correctly
			}
			double distBetweenTwoPoints = 0;
			double inverseDistSquared = 0;
			for( int col = 0; col < savedFeaturesMatrix.cols(); col++ ) {
				
				if( savedFeaturesMatrix.valueCount(col) == 0) {  // this means that this is a continuous attribute
					if( savedFeaturesMatrix.get(trainingInstance, col) == Double.MAX_VALUE ) { 		// check to see if it is missing value
						// then replace the difference with 1 (We could say that it is the (other value + 1) if that is easier )
						distBetweenTwoPoints += 1;
					} else {
						double differenceForOneVariable = ( savedFeaturesMatrix.get(trainingInstance, col) - featuresForPrediction[col] );
						distBetweenTwoPoints += (differenceForOneVariable * differenceForOneVariable );	
					}
				} else {  		// this means that this is a nominal attribute
					if( savedFeaturesMatrix.get(trainingInstance, col) == Double.MAX_VALUE ) { 		// check to see if it is missing value
						// then replace the difference with 1 (We could say that it is the (other value + 1) if that is easier )
						distBetweenTwoPoints += 1;
					} else {
						if( savedFeaturesMatrix.get(trainingInstance, col) == featuresForPrediction[col] ) {
							distBetweenTwoPoints += 0;
						} else {
							distBetweenTwoPoints += 1;
						}
					}
				}
			}
			inverseDistSquared = ( 1 / ( distBetweenTwoPoints + 0.0000001 ) );
			KNNPoint pointToInsert = distanceTable.get( counterOfRowWeAreInsertingAt );
			pointToInsert.distanceFromPointWereTryingToPredict = distBetweenTwoPoints;
			pointToInsert.trainingInstanceIndex = trainingInstance;
			pointToInsert.inverseDistanceSquared = inverseDistSquared;
			distanceTable.set( counterOfRowWeAreInsertingAt , pointToInsert );
			
			counterOfRowWeAreInsertingAt++;
		}
		return distanceTable;
	}
	
	
	/*
	 * Since we only want the k top values from the lookup table, it needs to be sorted from
	 * least to greatest
	 */
	private ArrayList<KNNPoint> sortTheDistanceLookupTable( ArrayList<KNNPoint> distanceTable) {
		Collections.sort( distanceTable, new Comparator<KNNPoint>() {
			@Override
			public int compare(KNNPoint element1, KNNPoint element2 ) {
				double knnPoint1Dist = (double)element1.distanceFromPointWereTryingToPredict;
				double knnPoint2Dist = (double)element2.distanceFromPointWereTryingToPredict;
				if( knnPoint1Dist < knnPoint2Dist ) return -1;
				if( knnPoint1Dist > knnPoint2Dist ) return 1;
				return 0;
			}
		});
		return distanceTable;
	}
	
	
	/*
	 * We look at the labels for the k-smallest distances in the distanceTable.
	 * We create a histogram for the labels, and return the label that is most common.
	 */
	private double extractLabelWithMostVotes( ArrayList<KNNPoint> distanceTable ) {
		if( useRegression == true ) {
			return computeRegression( distanceTable );
		}
		double labelToReturn = -1; // some arbitrary value
		double[] labelBuckets = new double[ savedLabelsMatrix.valueCount(0) ];
		for( int pointNumber = 0; pointNumber < kValue ; pointNumber++ ) {
			
			if( pointNumber >= numInstancesUsed) continue;
			
			KNNPoint oneOfKClosestPoints = distanceTable.get( pointNumber );
			int originalRowIndexInLabelsMatrix  = oneOfKClosestPoints.trainingInstanceIndex; // we only ever use the 0th column
			double labelAtThisPoint = savedLabelsMatrix.get(originalRowIndexInLabelsMatrix, 0 );
			if( useWeightedDistance == true) {
				labelBuckets[ (int)labelAtThisPoint ] += ( oneOfKClosestPoints.inverseDistanceSquared ) ;
				
			} else {
				labelBuckets[ (int)labelAtThisPoint ]++;
			}
		}
		double storedMax = Double.MIN_VALUE;
		for( int i = 0; i < savedLabelsMatrix.valueCount(0) ; i++ ) {
			if( labelBuckets[i] > storedMax ) {
				storedMax = labelBuckets[i];
				labelToReturn = i;
			}
		}
		return labelToReturn;
	}

	



	
	/*
	 * Compute Regression
	 * 
	 * UNWEIGHTED: Can do non-weighted regression by letting the output be mean of the k-nearest-neighbors
	 * 
	 * WEIGHTED:
	 * Can also do regression by letting the output be the weighted mean of the k-nearest-neighbors
	 * weight = inverse square of distance
	 * multiply weights by output value , add them all up.
	 * Then divide by sum of weights.
	 */
	public double computeRegression ( ArrayList<KNNPoint> distanceTable ) {
		
		if( useWeightedDistance == true ) { // WEIGHTED
			double weightedAverage = 0;
			double sumOfTheDistanceWeights = 0;
			for ( int k = 0; k < kValue; k++ ) {			
				double distBetweenTwoPoints = distanceTable.get(k).inverseDistanceSquared;
				sumOfTheDistanceWeights += distBetweenTwoPoints;
				weightedAverage += ( savedLabelsMatrix.get( (distanceTable.get(k).trainingInstanceIndex) , 0) * distBetweenTwoPoints );	
			}
			weightedAverage = (weightedAverage / sumOfTheDistanceWeights );
			return weightedAverage ;
		}
		// UNWEIGHTED
		double weightedAverage = 0;
		for ( int k = 0; k < kValue; k++ ) {			
			weightedAverage += ( savedLabelsMatrix.get( (distanceTable.get(k).trainingInstanceIndex) , 0) );	
		}
		weightedAverage = (weightedAverage / kValue ); 
		return weightedAverage ;
	}
	

/*
 * Elements cannot be inserted into ArrayLists until they are filled with
 *  instances at least up to the index you want to insert into.
 *  
 * Note that we only make the distance table as large as the number of instances
 * we will actually use (ie number will be smaller than original since we implement
 * reduction of training instances).
 * 
 * My implementation below allows me to dynamically change the size of the distanceTable
 * even during train() and also during predict(), since we start off saying that
 * numInstancesUsed is the full size, and then we gradually decrement it.
 */
	public ArrayList<KNNPoint> initializeDistanceTableWithZeroValues( ArrayList<KNNPoint> distanceTable ) {
		
			for( int row = 0; row < numInstancesUsed; row++ ) { 
				KNNPoint initializedPoint = new KNNPoint();
				distanceTable.add( row, initializedPoint );
			}
		return distanceTable;
	}
}







// if it gets it right, then delete it -- for classification
// 
