
/*
 * John Lambert
 * I implement K Means Clustering.
 */
	import java.text.DecimalFormat;
	import java.util.ArrayList;
	import java.util.Random;
	import java.lang.*;
	import java.math.RoundingMode;
import java.util.*;

public class KMeansWithToolkitLearner extends SupervisedLearner {
	
	public int kValue = 7;
	public double sseFromPreviousIteration;
	public boolean reachedConvergence;
	
	
	/*
	 * Make sure that you ignore the label column
	 */
	//public void train(Matrix features, Matrix labels) throws Exception {
	public void train(Matrix data, Matrix labels) throws Exception {
		
		sseFromPreviousIteration = Double.MAX_VALUE; // SOME ARBITRARY VALUE
		reachedConvergence = false;
		// make an array of all of the data points, or matrix or whatever
		// that is the Matrix "data"
		//Matrix data = new Matrix(features, 0, 1, features.rows(), features.cols() - 1 ); // labor data set's first col is useless
		int iterationCounter = 1;
		// clusters = we will make another array that shows the cluster number for the corresponding row in the matrix
		int[] assignedCentroids = new int[ data.rows() ];
		// For our centroid, choose the first k points in data set
		ArrayList<double[]> centroids = new ArrayList<double[]>();
		centroids = initializeTheCentroidsArrayList( centroids, data );
		while( reachedConvergence == false ) { // while we haven't converged yet
			
			System.out.println( "\n***************");
			System.out.println("Iteration " + iterationCounter);
			System.out.println("***************");
			System.out.println("Computing Centroids:" );
			// We assign all training instances to closest centroid
			printOutCentroidLocations( centroids , data);
			assignedCentroids = assignAllTrainingInstancesToClosestCentroid( centroids, data, assignedCentroids);
			printOutAssignments( assignedCentroids);
			centroids = calculateNewCentroidPositions( centroids, data, assignedCentroids );
			iterationCounter++;
		}
		System.out.println("\nSSE has converged\n");
		int[] histogram = printOutHowManyInstancesPerCluster( assignedCentroids );
		calculateSilhouetteScore( centroids, data , assignedCentroids, histogram);
	}

	
	
	/*
	 * We initialize ArrayList with first k points in data set, 
	 * or we initialize it with k random data points
	 */
	public ArrayList<double[]> initializeTheCentroidsArrayList( ArrayList<double[]> centroids, Matrix data ) {
		HashSet<Integer> myRandomDataPointsSet= new HashSet<Integer>();
		Random rand = new Random();
		for( int i = 0; i < kValue; i++ ) {
			int nextRandomDataPoint = rand.nextInt(data.rows());
			while( myRandomDataPointsSet.contains(nextRandomDataPoint) ) {
				nextRandomDataPoint = rand.nextInt(data.rows());
			}
			myRandomDataPointsSet.add( nextRandomDataPoint);
		}

		Iterator iterator = myRandomDataPointsSet.iterator();  // create an iterator
		      
		// check values
		while (iterator.hasNext() ) {
			int myInstanceNumber = (Integer)iterator.next();
			double[] randomDataPointsFeatureVector = new double[ data.cols() ];  
			for( int colOfFeatureVector = 0; colOfFeatureVector < data.cols() ; colOfFeatureVector++ ) {
				randomDataPointsFeatureVector[ colOfFeatureVector ] = data.get( myInstanceNumber , colOfFeatureVector);
			}
			centroids.add( randomDataPointsFeatureVector );
		}
		   
		
		//for( int centroidIndex = 0; centroidIndex < kValue; centroidIndex++ ) {
			//double[] centroidFeatureVector = new double[ data.cols() ];
			//for( int colOfFeatureVector = 0; colOfFeatureVector < data.cols() ; colOfFeatureVector++ ) {
//				centroidFeatureVector[ colOfFeatureVector ] = data.get( centroidIndex, colOfFeatureVector ); // if first k centroids are first k data points
			//}
			//centroids.add( centroidFeatureVector);
		//}
		return centroids;
	}
	
	
	
	
	/*
	 * This function...
	 * 
	 * For each data point, compute distance to each of those items in centroids array		
	 */
	public int[] assignAllTrainingInstancesToClosestCentroid( ArrayList<double[]> centroids, Matrix data, int[] assignedCentroids) {
		for( int instanceWeAreAssigning = 0; instanceWeAreAssigning < data.rows() ; instanceWeAreAssigning++ ) {
			double minDistanceBetweenTwoPoints = Double.MAX_VALUE;
			int closestCentroid = -1;
			for( int centroidIndex = 0; centroidIndex < kValue; centroidIndex++ ) {
				double distBetweenPointAndCentroid = 0;
				 // should I bother to take the square root, or is that unnecessary work because monotonically increasing
				for( int col = 0; col < data.cols(); col++ ) {
					if( data.valueCount(col) == 0) {  // this means that this is a continuous attribute
						if( data.get(instanceWeAreAssigning, col) == Double.MAX_VALUE ) { 		// check to see if it is missing value
							// then replace the difference with 1 (We could say that it is the (other value + 1) if that is easier )
							distBetweenPointAndCentroid += 1;
						} else {
							if( centroids.get( centroidIndex)[col] == Double.MAX_VALUE) { // the centroid had a missing value
								distBetweenPointAndCentroid += 1;
							} else {
								double differenceForOneVariable = ( data.get( instanceWeAreAssigning,col) - centroids.get(centroidIndex)[col] ); // or we can do Math.abs
								distBetweenPointAndCentroid += Math.abs(differenceForOneVariable );	
							}	
						}
					} else {  		// this means that this is a nominal attribute
						if( data.get(instanceWeAreAssigning, col) == Double.MAX_VALUE ) { 		// check to see if it is missing value
							// then replace the difference with 1 (We could say that it is the (other value + 1) if that is easier )
							distBetweenPointAndCentroid += 1;
						} else {
							if( data.get(instanceWeAreAssigning, col) == centroids.get(centroidIndex )[col] ) {
								distBetweenPointAndCentroid += 0;
							} else {
								distBetweenPointAndCentroid += 1;
							}
						}
					}
				}
				if( distBetweenPointAndCentroid < minDistanceBetweenTwoPoints ){
					minDistanceBetweenTwoPoints = distBetweenPointAndCentroid;
					closestCentroid = centroidIndex;
				}
			}
			assignedCentroids[instanceWeAreAssigning] = closestCentroid;
		}
		//use the old centroids here
		double sseOnAllCentroids = calculateSSEOnAllCentroids( centroids, data , assignedCentroids );
		sseFromPreviousIteration = sseOnAllCentroids; // update global variable
		//allows you to not bounce back and forth
		//if( (totalSSEForAllCentroids - previousTotalSSEForAllCentroids) < 0.1 ) {
		// we've converged
		// }
		return assignedCentroids;
	}
	
	
	/*
	 * Do I need assignedCentroids passed in here?
	 * 
	 * 	// use sum squared error to demonstrate that the points arent moving
		// have sse for each centroid
		// have total sse for all centroids
	
		// centroid is the target
		// sse is added all of that junk
		// sum sse for all centroids together. if not changing, we've converged
	 */
	public double calculateSSEOnAllCentroids ( ArrayList<double[]> centroids, Matrix data , int[] assignedCentroids ) {
		double sseForAllCentroids = 0;
		for( int centroidIndex = 0; centroidIndex < kValue; centroidIndex++ ) { // for each centroid
			double sseForThisCentroid = 0;
			for( int col = 0; col < data.cols(); col++ ) { //for each column
				for( int trainingInstance = 0; trainingInstance < data.rows(); trainingInstance++ ) { // for every data point in the centroid
					if( assignedCentroids[trainingInstance] == centroidIndex) {
						if( data.valueCount(col) == 0) {  // this means that this is a continuous attribute
							// add them all up, and take average
							if( (data.get(trainingInstance , col) == Double.MAX_VALUE) || (centroids.get(centroidIndex)[col] == Double.MAX_VALUE) ) { // we ignore missing values
								sseForThisCentroid += 1;
							} else {
								double diffBetweenThePoints = ( data.get( trainingInstance ,col) - centroids.get(centroidIndex)[ col ] );
								sseForThisCentroid += (diffBetweenThePoints * diffBetweenThePoints );
							}
						} else {  		// this means that this is a nominal attribute
							if( (data.get(trainingInstance , col) == Double.MAX_VALUE) || (centroids.get(centroidIndex)[ col ] == Double.MAX_VALUE) ) { // we ignore missing values
								sseForThisCentroid += 1;
							} else {
								if( data.get(trainingInstance , col) != centroids.get(centroidIndex )[col] ) {
									sseForThisCentroid += 1;
								}
							}
						}	
					}	
				}
			}
			System.out.println( "SSE for centroid # " + centroidIndex + " is " + sseForThisCentroid );
			sseForAllCentroids += sseForThisCentroid;
		}
		System.out.println( "SSE= " + sseForAllCentroids);
		if( ( Math.abs(sseFromPreviousIteration - sseForAllCentroids) < 0.01) ) {
			reachedConvergence = true;
		}
		return sseForAllCentroids;
	}
	
	
	
	/*
	 * This function...
	 */
	public ArrayList<double[]> calculateNewCentroidPositions( ArrayList<double[]> centroids, Matrix data, int[] assignedCentroids ) {
		// Find the average location of all of the data points assigned to that cluster
		for( int centroidIndex = 0; centroidIndex < kValue; centroidIndex++ ) {
			double[] newFeatureVectorForCentroid = new double[ data.cols() ];
			for( int col = 0; col < data.cols(); col++ ) { 
				if( data.valueCount(col) == 0) { // CONTINUOUS
					boolean weGotHere = false;
					int numDataPointsAssignedToThisCluster = 0;
					double averageForThisColumn = 0;
					for( int trainingInstance = 0; trainingInstance < data.rows(); trainingInstance++ ) {
						if( assignedCentroids[trainingInstance] == centroidIndex) {
							weGotHere = true;
							if( data.get(trainingInstance, col) != Double.MAX_VALUE ) {
								numDataPointsAssignedToThisCluster++;
								averageForThisColumn += data.get(trainingInstance, col);
							}
						}
					}
					if( (weGotHere == true) && (numDataPointsAssignedToThisCluster == 0) ) {  // ADD IN SPECIAL CASE IF THEY ARE ALL MISSING INSIDE OF THE COLUMN
						newFeatureVectorForCentroid[col] = Double.MAX_VALUE;// they were all zero
					} else {
						averageForThisColumn = (averageForThisColumn / numDataPointsAssignedToThisCluster);
						newFeatureVectorForCentroid[col] = averageForThisColumn;
					}
				} else { // NOMINAL
					boolean weGotHere = false;
					int[] histogram = new int[ data.cols() ]; // find most common, use histogram
					for( int trainingInstance = 0; trainingInstance < data.rows(); trainingInstance++ ) {
						if( assignedCentroids[trainingInstance] == centroidIndex) {
							weGotHere = true;
							if( data.get(trainingInstance, col) != Double.MAX_VALUE ) {
								histogram[ (int)data.get(trainingInstance, col) ]++;
							}
						}
					}

					int highestFrequency = Integer.MIN_VALUE;
					int mostFrequentValue = -1;
					for( int histogramIndex = 0; histogramIndex < histogram.length; histogramIndex++ ) {
						if( histogram[histogramIndex] > highestFrequency) {
							highestFrequency = histogram[histogramIndex];
							mostFrequentValue = histogramIndex;
						} else if( histogram[histogramIndex] == highestFrequency ) {
							if( mostFrequentValue > histogramIndex) { // higher in the metadata
								mostFrequentValue = histogramIndex; // break ties by whichever one comes first in the metadata list
							}
						}	
					}
					if( (weGotHere == true) && (highestFrequency == 0) ) {  // ADD IN SPECIAL CASE IF THEY ARE ALL MISSING INSIDE OF THE COLUMN
						newFeatureVectorForCentroid[col] = Double.MAX_VALUE;// they were all zero
					} else {
						newFeatureVectorForCentroid[col] = mostFrequentValue;
					}
				}
			}
			centroids.set( centroidIndex, newFeatureVectorForCentroid ); // make this the new centroid for that cluster	
		}
		return centroids;
	}
	
	/*
	 * For log spew comparison
	 */
	public void printOutCentroidLocations( ArrayList<double[]>centroids, Matrix data ) {
		DecimalFormat df = new DecimalFormat("#######.###");
		df.setRoundingMode(RoundingMode.CEILING);
		for( int i = 0; i < centroids.size(); i++ ) {
			System.out.print( "Centroid " + i +"= " );
			double[] array = centroids.get(i);
			for( int j = 0; j < array.length; j++ ) {
				//
				double value = array[j];
				if( value == Double.MAX_VALUE) {
					System.out.print("?, ");
				} else if( data.valueCount(j) == 0 ){ // continuous
					//System.out.print(df.format(array[j]) + ", ");
					System.out.printf( "%.3f" + ", ", array[j]);
					
				} else {
					System.out.printf( data.attrValue(j, (int)array[j]) + ", ");
				}
			}
			System.out.print("\n");
		}
	}
	
	/*
	 * For log spew comparison
	 */
	public void printOutAssignments( int[] assignedCentroids) {
		System.out.println( "Making Assignments");
		for( int trainingInstance = 0; trainingInstance < assignedCentroids.length; trainingInstance++ ) {
			if( (trainingInstance % 10) == 0 ) {
				System.out.print("\n");
			}
			System.out.print( trainingInstance + "=" + assignedCentroids[trainingInstance] + " ");
		}
	}


	public int[] printOutHowManyInstancesPerCluster( int[] assignedCentroids ) {
		int[] histogram = new int[ kValue ];
		for( int trainingInstance = 0; trainingInstance < assignedCentroids.length; trainingInstance++ ) {
			histogram[ assignedCentroids[trainingInstance] ]++;
		}
		for( int i = 0; i < histogram.length; i++ ) {
			System.out.println( "Centroid " + i + " has " + histogram[i] + " in it.");
		}
		return histogram;
	}

	
	
	public void calculateSilhouetteScore( ArrayList<double[]>centroids, Matrix data , int[] assignedCentroids,
			int[] histogram ) {
		double globalSilhouetteScore = 0;
		for( int centroidIndex = 0; centroidIndex < kValue; centroidIndex++ ) {
			double averageDistanceToCentroidFromPointsWithin = computeAverageDistanceToCentroidFromPointsWithin( centroids, data, assignedCentroids, histogram, centroidIndex );
			System.out.println( "Centroid #" + centroidIndex + " has a-value: " + averageDistanceToCentroidFromPointsWithin);
			// compactness a = the mean distance between a sample and all other points in the same class
			double avgDistFromThisCentroidToPointsInNextNearestCluster = computeAvgDistFromThisCentroidToPointsInNextNearestCluster ( 
					centroids, data, assignedCentroids, histogram, centroidIndex );
			// separability b = the mean distance between a sample and all other points in the next nearest cluster
			
			
			double silhouetteScore = ( (avgDistFromThisCentroidToPointsInNextNearestCluster - averageDistanceToCentroidFromPointsWithin ) / 
					Math.max(averageDistanceToCentroidFromPointsWithin, avgDistFromThisCentroidToPointsInNextNearestCluster) );
			System.out.println( "Centroid #" + centroidIndex + " has silhouette score: " + silhouetteScore);
			globalSilhouetteScore += silhouetteScore;
		}
		globalSilhouetteScore /= kValue;
		System.out.println( "GLOBAL SILHOUETTE SCORE IS: " + globalSilhouetteScore );
	}
	
	public double computeAvgDistFromThisCentroidToPointsInNextNearestCluster ( 
			ArrayList<double[]> centroids, Matrix data, int[] assignedCentroids, int[] histogram, int centroidIndex ) {
		double avgDistFromThisCentroidToPointsInNextNearestCluster = 0;
		
		double distToClosestCentroid = Double.MAX_VALUE;
		int closestCentroid = -1;
		for( int otherCentroidIndex = 0; otherCentroidIndex < kValue; otherCentroidIndex++ ) {
			if( otherCentroidIndex == centroidIndex) continue;
			double distToOtherCentroid = computeDistanceBetweenPoints( centroids.get(centroidIndex), centroids.get(otherCentroidIndex), data);
			if( distToOtherCentroid < distToClosestCentroid) {
				distToClosestCentroid = distToOtherCentroid;
				closestCentroid = otherCentroidIndex;
			}
		}
		//System.out.println( "Closest Centroid is This Far Away: " + distToClosestCentroid);
		
		for( int trainingInstance = 0; trainingInstance < assignedCentroids.length; trainingInstance++ ) {
			if( assignedCentroids[trainingInstance] == closestCentroid) {
				double[] thatInstancesFeatureVector = new double[ data.rows() ];
				for( int col = 0; col < data.cols(); col++ ) {
					thatInstancesFeatureVector[col] = data.get( trainingInstance, col);
				}
				double distFromMyCentroidToThatCentroidsPoint = computeDistanceBetweenPoints(  centroids.get( centroidIndex), thatInstancesFeatureVector, data);
				//System.out.println( "That point is this far away from my centroid: " + distFromMyCentroidToThatCentroidsPoint );
				avgDistFromThisCentroidToPointsInNextNearestCluster += distFromMyCentroidToThatCentroidsPoint;
			}
		}
		avgDistFromThisCentroidToPointsInNextNearestCluster /= (histogram[closestCentroid]);
		return avgDistFromThisCentroidToPointsInNextNearestCluster;
	}
	
	
	/*
	 * 
	 */
	public double computeDistanceBetweenPoints( double[] centroidOne, double[] centroidTwo, Matrix data ) {
		double distBetweenTwoCentroids = 0;
		// should I bother to take the square root, or is that unnecessary work because monotonically increasing
		for( int col = 0; col < data.cols(); col++ ) {
			if( data.valueCount(col) == 0) {  // this means that this is a continuous attribute
				if( centroidOne[col] == Double.MAX_VALUE ) { 		// check to see if it is missing value
					// then replace the difference with 1 (We could say that it is the (other value + 1) if that is easier )
					distBetweenTwoCentroids += 1;
				} else {
					if( centroidTwo[col] == Double.MAX_VALUE) { // the centroid had a missing value
						distBetweenTwoCentroids += 1;
					} else {
						double differenceForOneVariable = ( centroidOne[col] - centroidTwo[col] ); // or we can do Math.abs
						distBetweenTwoCentroids += Math.abs(differenceForOneVariable );	
					}	
				}
			} else {  		// this means that this is a nominal attribute
				if( centroidOne[col] == Double.MAX_VALUE ) { 		// check to see if it is missing value
					// then replace the difference with 1 (We could say that it is the (other value + 1) if that is easier )
					distBetweenTwoCentroids += 1;
				} else {
					if( centroidOne[col] == centroidTwo[col] ) {
						distBetweenTwoCentroids += 0;
					} else {
						distBetweenTwoCentroids += 1;
					}
				}
			}
		}
		return Math.sqrt(distBetweenTwoCentroids);
	}
	
	
	
	/*
	 * 
	 */
	public double computeAverageDistanceToCentroidFromPointsWithin( ArrayList<double[]>centroids, Matrix data , int[] assignedCentroids,
			int[] histogram, int centroidIndex ) {
		double averageDistanceToCentroidFromPointsWithin = 0;
		for( int trainingInstance = 0; trainingInstance < data.rows() ; trainingInstance++ ) {
			if( assignedCentroids[trainingInstance] == centroidIndex ) {
				double distBetweenPointAndCentroid = 0;
				// should I bother to take the square root, or is that unnecessary work because monotonically increasing
				for( int col = 0; col < data.cols(); col++ ) {
					if( data.valueCount(col) == 0) {  // this means that this is a continuous attribute
						if( data.get(trainingInstance, col) == Double.MAX_VALUE ) { 		// check to see if it is missing value
							// then replace the difference with 1 (We could say that it is the (other value + 1) if that is easier )
							distBetweenPointAndCentroid += 1;
						} else {
							if( centroids.get( centroidIndex)[col] == Double.MAX_VALUE) { // the centroid had a missing value
								distBetweenPointAndCentroid += 1;
							} else {
								double differenceForOneVariable = ( data.get( trainingInstance,col) - centroids.get(centroidIndex)[col] ); // or we can do Math.abs
								distBetweenPointAndCentroid += Math.abs(differenceForOneVariable );	
							}	
						}
					} else {  		// this means that this is a nominal attribute
						if( data.get(trainingInstance, col) == Double.MAX_VALUE ) { 		// check to see if it is missing value
							// then replace the difference with 1 (We could say that it is the (other value + 1) if that is easier )
							distBetweenPointAndCentroid += 1;
						} else {
							if( data.get(trainingInstance, col) == centroids.get(centroidIndex )[col] ) {
								distBetweenPointAndCentroid += 0;
							} else {
								distBetweenPointAndCentroid += 1;
							}
						}
					}
				}
				averageDistanceToCentroidFromPointsWithin += distBetweenPointAndCentroid;
			}
		}
		averageDistanceToCentroidFromPointsWithin /= ( histogram[centroidIndex] );
		return averageDistanceToCentroidFromPointsWithin;
	}
	
//Silhouette
// calculate with Euclidean
	// score with Eucliean
	// score with Manhattan
// calculat with Manhattan
	// score with Euclidean
	// score with Manhattan
	
	
	// We will not use predict
	public void predict(double[] features, double[] labels) throws Exception {
	}
}
