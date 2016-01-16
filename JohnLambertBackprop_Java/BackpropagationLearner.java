
/*
 * The BackpropagationLearner
 * 
 * My backpropagation algorithm includes the ability to create an arbitrary network structure (# of nodes, layers, etc.),
 * random weight initialization (small random weights with 0 mean), on-line/stochastic weight update, a reasonable stopping criterion, 
 * training set shuffling at each epoch, an option to include a momentum term, bias for hidden and output neurons.
 * 
 * The key to coding up a neural network is understanding how to represent its
 * intricate data structure.  Each layer in the network has its own matrix to represent
 *  ij interactions.  Thus, We need a data structure to hold each layer's matrix.
 * Since each layer's matrix can differ in dimensions from one another, it is
 * not suitable to use an array of arrays of arrays. We need an ArrayList of arrays of arrays
 *
 * We allow the user to specify the number of hidden nodes at each layer, along with the desired number of hidden layers
 * We also allow the user to specify the momentum term.
 * We hard-code these values in the Class' constructor
 * 
 * Note that with my Validation Set I do not stop the first epoch that the Validation Set does not get an improved accuracy.  
 * Rather, I keep track of the best solution so far (bssf) on the VS and consider a window of epochs (e.g. 5) and when there
 *  has been no improvement over bssf for the length of the window, then I stop the training.
 */

import java.util.ArrayList;
import java.util.Random;
import java.lang.*;
import java.util.*;

public class BackpropagationLearner extends SupervisedLearner {
	
	private int numHiddenLayers = 2 ;
	private int[] numNodesPerHiddenLayer = { 48, 48 } ; // number of elements in this array = numLayers
	private ArrayList<double[][]> arrayListOfEachLayersWeightMatrices;
	private ArrayList<double[][]> changeInWeightMatricesForEveryLayer;
	private ArrayList<double[][]> temporaryStashChangeInWeightMatricesForEveryLayer;
	private ArrayList<double[][]> previousChangeInWeightMatricesForEachLayer;
	private ArrayList<double[]> biasWeightsAcrossAllLayers;
	private ArrayList<double[]> changeInBiasArrayForEveryLayer;
	private ArrayList<double[]> previousBiasChangeInWeightsAcrossAllLayers;
	private ArrayList<double[]> temporarilyStashedChangeInBiasWeightsAcrossAllLayers;
	private ArrayList<double[]> arrayListOfEachLayersDeltaArray;
	private ArrayList<double[]> storedFNetForEachLayer; // f_net is the output that is fed into the next layer
		// we store f_net for each node
	private double learningRate = 0.1 ;
	private double momentum = 0.5;
	private double validationSetPercentageOfData = 0.80;
	private double[] globalStoredOutputNodeFNetValues;
	private double[] globalStoredOutputNodeTargetValues;

	/*
	 * We incorporate the ability to create an arbitrary network structure.
	 * We use array of arrays of doubles for each inter-layer matrix
	 * Thus, between each layer, we need a matrix of weights.
	 * Num rows * num columns in matrix = nodes in layer below * nodes in layer above
	 * 
	 * We use the Math library's pow function to raise to exponent: double pow(double base, double exponent)
	 * 
	 *                       Hidden Nodes in current Layer (j)
	 * previous layers nodes[                             ]
	 *             Features [         Wij                 ]
	 *                (i)   [                             ]
	 *       
	 * I set up a matrix with dimensions: [ nodes in previous layer ] [ nodes in next layer ]
	 *       
	 * Since we are traveling through one layer at a time, we need to have another data structure
	 * that will be outputs for this layer
	 * 
	 * I use for loops to initialize array of arrays ( allocated necessary memory)
	 * Please note that: number of layers + 1 = number of weight arrays needed
	 */
	public void train(Matrix features, Matrix labels) throws Exception {
		
		double[] recentAccuracies = new double[5];
		int currentAccuracyIndex = 0; 
		double currentAccuracy = 0;
		
		Random rand = new Random();
		// SHUFFLE labels, features together
		features.shuffle(rand, labels);
		
		// need to map 0,1, or 2 to the three dimensional vectors, DO N-OF-K-ENCODING FOR THE BACKPROPAGATION
		Matrix newNOfKLabelsMatrix = new Matrix();
		newNOfKLabelsMatrix.setSize( labels.rows(), labels.valueCount(0) ); // I HARD CODE IN THAT THERE SHOULD BE 3 OUTPUT NODES
		for( int row = 0; row < newNOfKLabelsMatrix.rows(); row++) { // for each instance
			for( int k = 0; k < labels.valueCount(0); k++ ) {
				if( labels.get(row, 0) == k) {
					for( int m = 0; m < labels.valueCount(0); m++ ) {
						newNOfKLabelsMatrix.set( row, m, 0 );
					}
					newNOfKLabelsMatrix.set( row, k, 1 );
				}	
			}
		}
		labels = newNOfKLabelsMatrix;
		
		// IMMEDIATELY SAVE SOME OF THIS, NEVER WILL TRAIN ON THESE
		// STICK THESE INTO A VALIDATION SET
		// ONCE MSE STARTS TO INCREASE AGAIN ON THE VALIDATION SET, WE'VE GONE TOO FAR
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
		
		// LOOP THROUGH AND PUT MOST OF FEATURES INTO featuresForTrainingTrimmed
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
		// LOOP THROUGH AND PUT LEFTOVER PORTION OF FEATURES INTO validationSet
		arrayListOfEachLayersWeightMatrices = new ArrayList<double[][]>();
		
		for( int i = 0; i < numHiddenLayers + 1 ; i++ ) { // each layer
			double[][] specificLayersWeightMatrix;
			if( i == 0 ) { // first hidden layer (Each layer owns its own weights)
				specificLayersWeightMatrix = new double[ features.cols() ][ numNodesPerHiddenLayer[i] ] ; // INPUTS are the rows
			} else if( i == numHiddenLayers) {
				specificLayersWeightMatrix = new double[ numNodesPerHiddenLayer[ i-1 ] ][ labels.cols() ] ; // OUTPUTS ARE THE COLUMNS
			} else {
				specificLayersWeightMatrix = new double[ numNodesPerHiddenLayer[i - 1] ][ numNodesPerHiddenLayer[ i ] ] ;
			}
			arrayListOfEachLayersWeightMatrices.add(specificLayersWeightMatrix ) ;
		}
		
		
		
		changeInWeightMatricesForEveryLayer = new ArrayList<double[][]>();
		
		for( int i = 0; i < numHiddenLayers + 1 ; i++ ) { // each layer
			double[][] specificLayersWeightMatrix;
			if( i == 0 ) { // first hidden layer (Each layer owns its own weights)
				specificLayersWeightMatrix = new double[ features.cols() ][ numNodesPerHiddenLayer[i] ] ; // INPUTS are the rows
			} else if( i == numHiddenLayers) {
				specificLayersWeightMatrix = new double[ numNodesPerHiddenLayer[ i-1 ] ][ labels.cols() ] ; // OUTPUTS ARE THE COLUMNS
			} else {
				specificLayersWeightMatrix = new double[ numNodesPerHiddenLayer[i - 1] ][ numNodesPerHiddenLayer[ i ] ] ;
			}
			changeInWeightMatricesForEveryLayer.add(specificLayersWeightMatrix ) ;
		}
		
		
		// allocate space/ initialize the previous change in weights that we'll use for momentum
		temporaryStashChangeInWeightMatricesForEveryLayer = new ArrayList<double[][]>();
		
		for( int i = 0; i < numHiddenLayers + 1 ; i++ ) { // each layer
			double[][] specificLayersWeightMatrix;
			if( i == 0 ) { // first hidden layer (Each layer owns its own weights)
				specificLayersWeightMatrix = new double[ features.cols() ][ numNodesPerHiddenLayer[i] ] ; // INPUTS are the rows
			} else if( i == numHiddenLayers) {
				specificLayersWeightMatrix = new double[ numNodesPerHiddenLayer[ i-1 ] ][ labels.cols() ] ; // OUTPUTS ARE THE COLUMNS
			} else {
				specificLayersWeightMatrix = new double[ numNodesPerHiddenLayer[i - 1] ][ numNodesPerHiddenLayer[ i ] ] ;
			}
			temporaryStashChangeInWeightMatricesForEveryLayer.add(specificLayersWeightMatrix ) ;
		}
		
		// ALLOCATE SPACE FOR DELTA ( INTERMEDIATE VALUES THAT WE USE TO UPDATE THE WEIGHTS)
		
		arrayListOfEachLayersDeltaArray = new ArrayList<double[]>();
		//  EACH LAYER HAS AN ARRAY OF DELTA VALUES
		for( int i = 0; i < numHiddenLayers + 2 ; i++ ) { // each layer  // OF COURSE WE COULD HAVE DONE numHiddenLayers + 1, but I want consistency with fnet ArrayList
			double[] specificLayersDeltaArray;
			if( i == 0 ) { // first hidden layer (Each layer owns its own weights)
				specificLayersDeltaArray = new double[ features.cols() ]; // INPUTS are the rows
			} else if( i == (numHiddenLayers + 1) ) {
				//specificLayersDeltaArray = new double[ numNodesPerHiddenLayer[ i-1 ] ]  ; //[ numNodesPerHiddenLayer[ labels.cols() ] ] ; // OUTPUTS ARE THE COLUMNS
				specificLayersDeltaArray = new double[ labels.cols() ]; // FIND OUT # NODES AT EACH LEVEL
			} else {
				specificLayersDeltaArray = new double[ numNodesPerHiddenLayer[i - 1] ] ; 
			}
			arrayListOfEachLayersDeltaArray.add( specificLayersDeltaArray ) ;
		}
		
		previousChangeInWeightMatricesForEachLayer = new ArrayList<double[][]>();
		
		for( int i = 0; i < numHiddenLayers + 1 ; i++ ) { // each layer
			double[][] specificLayersWeightMatrix;
			if( i == 0 ) { // first hidden layer (Each layer owns its own weights)
				specificLayersWeightMatrix = new double[ features.cols() ][ numNodesPerHiddenLayer[i] ] ; // INPUTS are the rows
			} else if( i == numHiddenLayers) {
				specificLayersWeightMatrix = new double[ numNodesPerHiddenLayer[ i-1 ] ][ labels.cols() ] ; // OUTPUTS ARE THE COLUMNS
			} else {
				specificLayersWeightMatrix = new double[ numNodesPerHiddenLayer[i - 1] ][ numNodesPerHiddenLayer[ i ] ] ;
			}
			previousChangeInWeightMatricesForEachLayer.add(specificLayersWeightMatrix ) ;
		}
		
		// INITIALIZE ALL OF PREVIOUS DELTA VALUES TO 0 [ THIS IS DONE AUTOMATICALLY, CAN DELETE ALL OF THIS CODE ]
		
		// initialize all weights randomly ( small random weights with 0 mean)
		
		double[][] currentLayersWeightMatrix;
		for( int i = 0; i< numNodesPerHiddenLayer.length + 1 ; i++ ) { // scroll across each layer
		
			currentLayersWeightMatrix = arrayListOfEachLayersWeightMatrices.get(i) ;
			for( int j = 0; j < currentLayersWeightMatrix.length ; j++ ) {
				for( int k = 0; k < currentLayersWeightMatrix[j].length; k++ ) {
					currentLayersWeightMatrix[j][k] =  ( 2 * rand.nextDouble() ) - 1 ;
				}
			}
		}
		
// GO THROUGH AND ADD THE SPECIFIC WEIGHTS
//Initial Weights:
		

		
		// PUT ALL BIAS WEIGHTS INTO ARRAYLIST (ONE ARRAY FOR EACH LAYER'S BIAS WEIGHTS)
		biasWeightsAcrossAllLayers = new ArrayList<double[]>();
		for( int i = 0; i< numHiddenLayers + 1; i++) {
			if( i < numHiddenLayers) {
				double[] biasArrayToBeAdded = new double[ numNodesPerHiddenLayer[i] ];
				biasWeightsAcrossAllLayers.add( biasArrayToBeAdded );
			} else {
				double[] biasArrayForOutputNodesToBeAdded = new double[ labels.cols() ];
				biasWeightsAcrossAllLayers.add( biasArrayForOutputNodesToBeAdded );
			}
		}
		
		
		
		double[] currentBiasLayersWeightArray;
		for( int i = 0; i< numNodesPerHiddenLayer.length + 1 ; i++ ) { // scroll across each layer
			currentBiasLayersWeightArray = biasWeightsAcrossAllLayers.get(i) ;
			for( int j = 0; j < currentBiasLayersWeightArray.length ; j++ ) {
				
				currentBiasLayersWeightArray[j] =  ( 2 * rand.nextDouble() ) - 1 ;
			}
		}
		
		//We'll need to store the previous bias weights
		previousBiasChangeInWeightsAcrossAllLayers = new ArrayList<double[]>();
		for( int i = 0; i< numHiddenLayers + 1; i++) {
			if( i < numHiddenLayers) {
				double[] biasArrayToBeAdded = new double[ numNodesPerHiddenLayer[i] ];
				previousBiasChangeInWeightsAcrossAllLayers.add( biasArrayToBeAdded );
			} else {
				double[] biasArrayForOutputNodesToBeAdded = new double[ labels.cols() ];
				previousBiasChangeInWeightsAcrossAllLayers.add( biasArrayForOutputNodesToBeAdded );
			}
		}
		
		// temporarily stashed bias weights across all layers
		temporarilyStashedChangeInBiasWeightsAcrossAllLayers = new ArrayList<double[]>();
		for( int i = 0; i< numHiddenLayers + 1; i++) {
			if( i < numHiddenLayers) {
				double[] biasArrayToBeAdded = new double[ numNodesPerHiddenLayer[i] ];
				temporarilyStashedChangeInBiasWeightsAcrossAllLayers.add( biasArrayToBeAdded );
			} else {
				double[] biasArrayForOutputNodesToBeAdded = new double[ labels.cols() ];
				temporarilyStashedChangeInBiasWeightsAcrossAllLayers.add( biasArrayForOutputNodesToBeAdded );
			}
		}
		
		changeInBiasArrayForEveryLayer = new ArrayList<double[]>();
		for( int i = 0; i< numHiddenLayers + 1; i++) {
			if( i < numHiddenLayers) {
				double[] biasArrayToBeAdded = new double[ numNodesPerHiddenLayer[i] ];
				changeInBiasArrayForEveryLayer.add( biasArrayToBeAdded );
			} else {
				double[] biasArrayForOutputNodesToBeAdded = new double[ labels.cols() ];
				changeInBiasArrayForEveryLayer.add( biasArrayForOutputNodesToBeAdded );
			}
		}

		
		// INITIALIZE BIAS FOR HIDDEN AND OUTPUT NEURONS
		
		// Stochastic weight update
		// SOMEHOW GOT TO INITIALIZE ALL OF THIS, ADD BLANKS, SO THAT LATER WE CAN storedFNetForEachLayer.set( i, blah );
		
		storedFNetForEachLayer = new ArrayList<double[]>(); // f_net is the output that is fed into the next layer
		for( int i = 0; i < numHiddenLayers + 2; i++ ) { // WE HAVE ONE MORE layer of fnet( consider inputs as fnet)
			double[] thisLayersFNetValues;
			// COULD DO IF/ELSE STATEMENTS IF WE ARE LOOKING AT INPUTS, OR THEN HIDDEN NODES,
			if( i == 0) {
				thisLayersFNetValues = new double[ features.cols() ]; // FIND OUT # NODES AT EACH LEVEL
			} else if (i == numHiddenLayers + 1 ) { // OR IS IT +1
				thisLayersFNetValues = new double[ labels.cols() ]; // FIND OUT # NODES AT EACH LEVEL
			} else {
				thisLayersFNetValues = new double[ numNodesPerHiddenLayer[ i-1 ] ]; // FIND OUT # NODES AT EACH LEVEL
			}
			storedFNetForEachLayer.add( thisLayersFNetValues );
		}
		
		
//-----BEGIN THE TRAINING-----
		double netValAtNode = 0;
		double fOfNetValAtNode = 0;
		for( int epoch = 0; epoch < 10000 ; epoch++ ) { // For each epoch, cap it at 10000, we want to avoid infinite loop
			System.out.println("---Epoch " + epoch + "---");
			for ( int instance = 0; instance < features.rows() ; instance++ ) { // later we will swap this Matrix for featuresForTrainingTrimmed
// GO FORWARD ---------------------------------------------------------------------------------------------------------------------
//				System.out.println("Forward propagating...");
				for( int layer = 0; layer < numHiddenLayers + 2 ; layer++ ) { // HERE LAYER DENOTES HIDDEN LAYER
					if( layer == 0) {
						storedFNetForEachLayer.set( layer,  Arrays.copyOf( features.row(instance) , features.row(0).length ) );
						continue;
					}
					double[] thisLayersFNetValues = storedFNetForEachLayer.get(layer); 	// make a new array of doubles  CAN I PLEASE DELETE THIS LINE OF CODE
					for( int node = 0; node < storedFNetForEachLayer.get(layer).length ; node++ ) {
						netValAtNode = 0;						
						// FIND THE CROSS PRODUCT;
						// use a for loop to multiply each col of weights vector by each col of outputsFromPreviousLayer
						for( int colInInputVector = 0; colInInputVector< storedFNetForEachLayer.get(layer-1).length ; colInInputVector++ ) {					
							netValAtNode +=  ( storedFNetForEachLayer.get(layer-1)[colInInputVector] *  arrayListOfEachLayersWeightMatrices.get(layer-1)[colInInputVector ][ node  ]) ;
						}
						netValAtNode += ( biasWeightsAcrossAllLayers.get(layer-1)[ node ] );
						if( netValAtNode < 0) { // make special function
							fOfNetValAtNode = (1 / (1 + Math.pow( Math.E, (-1 * netValAtNode ) ) ) );							
						} else { // normal
							fOfNetValAtNode = (1 / (1 + (1 / (Math.pow( Math.E, ( netValAtNode ) ) ) ) ) ); // if it was positive, then we raise to neg exponent
						}
						thisLayersFNetValues[node] = fOfNetValAtNode; 	// stick it into the object
					}
					storedFNetForEachLayer.set( layer, thisLayersFNetValues ); // or if we are editing object, this is not even necessary DOUBLE CHECK
				}
// ---NOW FOR THIS INSTANCE, GO BACKWARDS-----------------------------------------------------------------------------------------------------------------------
				//System.out.println("Back propagating...");
				// UPDATE THE WEIGHTS
				for( int layer = numHiddenLayers + 1; layer > 0; layer-- ) { // ACROSS EACH LAYER BACKWARD 
					if( layer == numHiddenLayers + 1) { // THIS IS AN OUTPUT LAYER
						for( int node = 0; node < labels.cols(); node++ ) {
							double deltaArrayForThisLayer[] = arrayListOfEachLayersDeltaArray.get(layer);
							deltaArrayForThisLayer[node] =(  ( labels.get(instance, node) - storedFNetForEachLayer.get(layer)[node] ) * (storedFNetForEachLayer.get(layer)[node]) * 
								(1 - (storedFNetForEachLayer.get(layer)[node]) ) );
							// should automatically be set since we get the objects address from heap memory, and change it
							for( int inputToThisNode = 0; inputToThisNode < numNodesPerHiddenLayer[layer-2] + 1 ; inputToThisNode++ ) {
								double changeInWeightBetweenIJ = 0;
								if( inputToThisNode == numNodesPerHiddenLayer[layer-2] ) { // this is a bias node
									
									changeInWeightBetweenIJ = ( learningRate * 1 * arrayListOfEachLayersDeltaArray.get(layer)[node] ); // NEED TO ADD STUFF FOR MOMENTUM
									double[] thisLayersBiasWeights = changeInBiasArrayForEveryLayer.get(layer-1); // NEED TO ADD STUFF FOR MOMENTUM
									thisLayersBiasWeights[node] = ( changeInWeightBetweenIJ); // NEED TO ADD STUFF FOR MOMENTUM
								} else {
								
									changeInWeightBetweenIJ = ( learningRate * storedFNetForEachLayer.get(layer-1)[inputToThisNode] * arrayListOfEachLayersDeltaArray.get(layer)[node]);
									//double[][] thisLayersWeightMatrix = arrayListOfEachLayersWeightMatrices.get(layer-1);
									//thisLayersWeightMatrix[inputToThisNode][node] += ( changeInWeightBetweenIJ );
									double[][] changeInWeightsMatrixForThisLayer = changeInWeightMatricesForEveryLayer.get(layer-1);
									changeInWeightsMatrixForThisLayer[inputToThisNode][node] = changeInWeightBetweenIJ;
								}
							}
						}
					} else {
						
						for( int node = 0; node < numNodesPerHiddenLayer[layer-1] + 1 ; node++ ) {  // ACROSS EACH HIDDEN LAYER (ie these are not output nodes)
							double deltaArrayForThisLayer[] = arrayListOfEachLayersDeltaArray.get(layer);
							
							if( node == numNodesPerHiddenLayer[layer-1] ) { // this is a bias node
								// change in weight = learningRate * 
							} else {  // this is not a bias node
								double summedOutgoingWeightsCrossOutputDelta = 0;
								
								for( int outgoingEdgeToOutgoingNode = 0; outgoingEdgeToOutgoingNode< arrayListOfEachLayersDeltaArray.get(layer+1).length ; outgoingEdgeToOutgoingNode++ ) {					
									summedOutgoingWeightsCrossOutputDelta +=  ( arrayListOfEachLayersDeltaArray.get(layer+1)[outgoingEdgeToOutgoingNode] *  
											arrayListOfEachLayersWeightMatrices.get(layer)[ node ][outgoingEdgeToOutgoingNode ]) ;
									
								}
								
								deltaArrayForThisLayer[node] =( ( summedOutgoingWeightsCrossOutputDelta ) * (storedFNetForEachLayer.get(layer)[node]) * 
									(1 - (storedFNetForEachLayer.get(layer)[node]) ) );
								// should automatically be set since we get the objects address from heap memory, and change it
								
								if( layer == 1) {
									// need a for loop across the neural net's input nodes
									for( int inputToTheNeuralNet = 0; inputToTheNeuralNet < features.cols() + 1; inputToTheNeuralNet++ ) {
										double changeInWeightBetweenIJ = 0;
										if( inputToTheNeuralNet == features.cols() ) { // then we know that this is our bias node
											
											changeInWeightBetweenIJ = ( learningRate * 1 * arrayListOfEachLayersDeltaArray.get(layer)[node] ); // NEED TO ADD STUFF FOR MOMENTUM
											double[] thisLayersBiasWeights = changeInBiasArrayForEveryLayer.get(layer-1); // NEED TO ADD STUFF FOR MOMENTUM
											thisLayersBiasWeights[node] = ( changeInWeightBetweenIJ); // NEED TO ADD STUFF FOR MOMENTUM
											
										} else {
											
											changeInWeightBetweenIJ = ( learningRate * storedFNetForEachLayer.get(layer-1)[inputToTheNeuralNet] * arrayListOfEachLayersDeltaArray.get(layer)[node]);
											double[][] changeInWeightsMatrixForThisLayer = changeInWeightMatricesForEveryLayer.get(layer-1);
											changeInWeightsMatrixForThisLayer[inputToTheNeuralNet][node] = changeInWeightBetweenIJ;
										}
									}
								} else {
									for( int inputToThisNode = 0; inputToThisNode < numNodesPerHiddenLayer[layer-2] + 1 ; inputToThisNode++ ) {
										double changeInWeightBetweenIJ = 0;
										if( inputToThisNode == numNodesPerHiddenLayer[layer-2] ) { // this is a bias node
											
											changeInWeightBetweenIJ = ( learningRate * 1 * arrayListOfEachLayersDeltaArray.get(layer)[node] ); // NEED TO ADD STUFF FOR MOMENTUM
											double[] thisLayersBiasWeights = changeInBiasArrayForEveryLayer.get(layer-1); // NEED TO ADD STUFF FOR MOMENTUM
											thisLayersBiasWeights[node] = ( changeInWeightBetweenIJ); // NEED TO ADD STUFF FOR MOMENTUM
											
										} else {
											
											changeInWeightBetweenIJ = ( learningRate * storedFNetForEachLayer.get(layer-1)[inputToThisNode] * arrayListOfEachLayersDeltaArray.get(layer)[node]);
											//double[][] thisLayersWeightMatrix = arrayListOfEachLayersWeightMatrices.get(layer-1);
											//thisLayersWeightMatrix[inputToThisNode][node] += ( changeInWeightBetweenIJ );
											double[][] changeInWeightsMatrixForThisLayer = changeInWeightMatricesForEveryLayer.get(layer-1);
											changeInWeightsMatrixForThisLayer[inputToThisNode][node] = changeInWeightBetweenIJ;
										}
									}
								}
							}
						}
					}					
				}	
				
//				System.out.printf( "e_0=%.17f,  e_1=%.17f, e_2=%.17f, e_3=%.17f\n" , arrayListOfEachLayersDeltaArray.get(2)[0], arrayListOfEachLayersDeltaArray.get(1)[0] , 
//				arrayListOfEachLayersDeltaArray.get(1)[1] , arrayListOfEachLayersDeltaArray.get(1)[2]);				
//				System.out.println("Descending Gradient...");
				
				
				
//				// PUT TEMPORARILY STASHED INTO PREVIOUS
//				// ONLY HERE SHOULD WE PUT IN THE STASHED WEIGHTS INTO THE PREVIOUS-STASH-SPOT
//				// PUT STASHED INTO PREVIOUS
//				
//				// update the bias weights
				
				// GET NEW CHANGE IN WEIGHT THANKS TO MOMENTUM, PLACE IN PREVIOUS SPOT
				// should be changeInBiasArrayForEveryLayer not 
				
				for( int w = 0; w < previousBiasChangeInWeightsAcrossAllLayers.size(); w++ ) {
					for( int y = 0; y < previousBiasChangeInWeightsAcrossAllLayers.get(w).length; y++ ) {
						double currentChangeInWeightVal = changeInBiasArrayForEveryLayer.get(w)[y];
						double[] fullBiasWeightList = biasWeightsAcrossAllLayers.get(w);
						double previousXYCoordInBiasWeightMatrix = previousBiasChangeInWeightsAcrossAllLayers.get(w)[y];
						double thisIsTheWeightChangeIncludingMomentum = ( currentChangeInWeightVal + (momentum*previousXYCoordInBiasWeightMatrix)) ;
						fullBiasWeightList[y] += thisIsTheWeightChangeIncludingMomentum;
						double[] arrayOfPreviousBiases = previousBiasChangeInWeightsAcrossAllLayers.get(w);
						arrayOfPreviousBiases[y] = thisIsTheWeightChangeIncludingMomentum;
					}
				} 
				
				// GET NEW CHANGE IN WEIGHT THANKS TO MOMENTUM, PLACE IN PREVIOUS SPOT
				
				// We update the weights ( by adding the changes in weights to the weight matrices) after every layer has been processed
				for( int w = 0; w < arrayListOfEachLayersWeightMatrices.size(); w++ ) {
					for( int y = 0; y < arrayListOfEachLayersWeightMatrices.get(w).length; y++ ) {
						for( int z = 0; z < arrayListOfEachLayersWeightMatrices.get(w)[y].length ; z++ ) {
							double currentXYCoordInMatrix = changeInWeightMatricesForEveryLayer.get(w)[y][z];
							double[] fullWeightListForLayer = arrayListOfEachLayersWeightMatrices.get(w)[y];
							
							double previousXYCoordInChangeInWeightMatrix = previousChangeInWeightMatricesForEachLayer.get(w)[y][z];
							double thisIsTheWeightChangeIncludingMomentum = (currentXYCoordInMatrix + ( previousXYCoordInChangeInWeightMatrix*momentum) );
							fullWeightListForLayer[z] += thisIsTheWeightChangeIncludingMomentum;
							double[][] arrayOfPreviousBiases = previousChangeInWeightMatricesForEachLayer.get(w);
							arrayOfPreviousBiases[y][z] = thisIsTheWeightChangeIncludingMomentum;
							//newWeight(at next round t+1) = learningRate * delta_at_node_we_feed_into * Xi + momentum_parameter * change_in_weight_at_t
							// momentum goes into the weight updates ( not in the change in weights)							
						}
					}
				} 
				
				
				
				
//				System.out.printf( "w_0=%.17f,  w_1=%.17f, w_2=%.17f, w_3=%.17f, w_4=%.17f, w_5=%.17f,\n w_6=%.17f, w_7=%.17f, w_8=%.17f, w_9=%.17f," +
//						"w_10=%.17f, w_11=%.17f,\n w_12=%.17f\n" , 
//						biasWeightsAcrossAllLayers.get(1)[0], arrayListOfEachLayersWeightMatrices.get(1)[0][0] , 
//						arrayListOfEachLayersWeightMatrices.get(1)[1][0] , arrayListOfEachLayersWeightMatrices.get(1)[2][0] , biasWeightsAcrossAllLayers.get(0)[0],
//				arrayListOfEachLayersWeightMatrices.get(0)[0][0], arrayListOfEachLayersWeightMatrices.get(0)[1][0], biasWeightsAcrossAllLayers.get(0)[1],
//				arrayListOfEachLayersWeightMatrices.get(0)[0][1], arrayListOfEachLayersWeightMatrices.get(0)[1][1], arrayListOfEachLayersWeightMatrices.get(0)[0][2],
//				biasWeightsAcrossAllLayers.get(0)[2], arrayListOfEachLayersWeightMatrices.get(0)[0][2],  arrayListOfEachLayersWeightMatrices.get(0)[1][2]);		
//				// ONLY AFTER THIS POINT HAS EVERY LAYER BEEN PROCESSED
				
			}
			
			//if( STOPPING CRITERIA MET ) {  // HAVE TO USE THE VALIDATION SET THIS TIME FOR THE STOPPING CRITERION
			currentAccuracy = calculateMSEOnValidationSet( featuresValidationSet , labelsValidationSet );
			//currentAccuracy = calculateMSEOnValidationSet( features , labels ); // On the training set now
			System.out.println(" Current MSE on epoch # " + epoch + " is: " + currentAccuracy ); 
			currentAccuracyIndex++;
			recentAccuracies[ currentAccuracyIndex % 5 ] = currentAccuracy;
			double sumAccuracies = 0;
			if( currentAccuracyIndex > 5) {
				for( int i=0; i< recentAccuracies.length ; i++ ) {
					sumAccuracies += Math.abs( recentAccuracies[ currentAccuracyIndex % 5 ] - recentAccuracies[i] );
				}
				if( sumAccuracies < 0.01 ) { // we only stop training when measureAccuracy after 5 epochs does not increase by 0.01
					break ;
				}
			}

			// In theory, it would be wise here to go back to the old best weights because now we're already overfitting if the stopping criterion is met
			features.shuffle(rand, labels); // MUST SHUFFLE DATA ROWS AFTER EACH EPOCH,labels is the buddy matrix
		}
		return;
	}
		
	
	/*
	 * We feed in 
	 */
	public void predict(double[] features, double[] labels) throws Exception {
		
		double netValAtNode = 0;
		double fOfNetValAtNode = 0;
		// ALL I HAVE TO DO IS SEND IT FORWARD THROUGH THE NETWORK
		// only get one instance at a time
		
		for( int layer = 0; layer < numHiddenLayers + 2 ; layer++ ) { // HERE LAYER DENOTES HIDDEN LAYER
			if( layer == 0) {
				storedFNetForEachLayer.set( layer,  Arrays.copyOf( features , features.length ) );
				continue;
			}
			double[] thisLayersFNetValues = storedFNetForEachLayer.get(layer); 	// make a new array of doubles  CAN I PLEASE DELETE THIS LINE OF CODE
			for( int node = 0; node < storedFNetForEachLayer.get(layer).length ; node++ ) {
				netValAtNode = 0;						
				// FIND THE CROSS PRODUCT;
				// use a for loop to multiply each col of weights vector by each col of outputsFromPreviousLayer
				for( int colInInputVector = 0; colInInputVector< storedFNetForEachLayer.get(layer-1).length ; colInInputVector++ ) {					
					netValAtNode +=  ( storedFNetForEachLayer.get(layer-1)[colInInputVector] *  arrayListOfEachLayersWeightMatrices.get(layer-1)[colInInputVector ][ node  ]) ;
				}
				netValAtNode += ( biasWeightsAcrossAllLayers.get(layer-1)[ node ] );
				if( netValAtNode < 0) { // make special function
					fOfNetValAtNode = (1 / (1 + Math.pow( Math.E, (-1 * netValAtNode ) ) ) );							
				} else { // normal
					fOfNetValAtNode = (1 / (1 + (1 / (Math.pow( Math.E, ( netValAtNode ) ) ) ) ) ); // if it was positive, then we raise to neg exponent
				}
				thisLayersFNetValues[node] = fOfNetValAtNode; 	// stick it into the object
				
			}
			storedFNetForEachLayer.set( layer, thisLayersFNetValues ); // or if we are editing object, this is not even necessary DOUBLE CHECK
		}
		
		// pick the output that the network says it is
		// return it	
		
		// put what is beyond the hidden nodes into the labels matrix
		double maxPredictedFOfNetVal = -999999;
		int predictedClass = 0;
		double[] storedOutputNodeFNetValues = new double[ storedFNetForEachLayer.get(numHiddenLayers + 1 ).length ];
		double[] storedOutputNodeTargetValues = new double[ storedFNetForEachLayer.get(numHiddenLayers + 1 ).length ];
		
		for(int i = 0; i < storedFNetForEachLayer.get(numHiddenLayers + 1 ).length ; i++ ) {
			if( labels.length > 1) {
				storedOutputNodeFNetValues[i] = storedFNetForEachLayer.get(numHiddenLayers + 1 )[i];
				storedOutputNodeTargetValues[i] = labels[i];  			// time to go ahead and save what we had at each output node
			}
			if( storedFNetForEachLayer.get(numHiddenLayers + 1 )[i] > maxPredictedFOfNetVal ) {
				predictedClass = i;
				maxPredictedFOfNetVal = storedFNetForEachLayer.get(numHiddenLayers + 1 )[i];
			}
		}
		labels[0] = predictedClass;
		globalStoredOutputNodeFNetValues = storedOutputNodeFNetValues;
		globalStoredOutputNodeTargetValues = storedOutputNodeTargetValues;

		// labels is not expecting 0,1,0 IT IS EXPECTING 0,1,2
	}
	
	
	double calculateMSEOnValidationSet(Matrix featuresValidationSet, Matrix labelsValidationSet ) throws Exception {
		
		double sumSquaredError = 0;
		
		for( int instance = 0; instance < featuresValidationSet.rows() ; instance++ ) {
			double errorAcrossOutputNodes = 0;
			double[] predictedLabel = labelsValidationSet.row( instance); // this is the target
			predict( featuresValidationSet.row(instance), labelsValidationSet.row(instance) );
			for( int col = 0; col < globalStoredOutputNodeFNetValues.length; col++ ) {
				errorAcrossOutputNodes += ( globalStoredOutputNodeTargetValues[col] - globalStoredOutputNodeFNetValues[col] );
			}
			sumSquaredError += (errorAcrossOutputNodes * errorAcrossOutputNodes );
		}
		
		double MSE = ( sumSquaredError / (featuresValidationSet.rows() * globalStoredOutputNodeFNetValues.length) );
		
		return MSE;
	}
}







	
	