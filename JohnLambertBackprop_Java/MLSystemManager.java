// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------


import java.util.ArrayList;
import java.util.Random;
import java.io.File;


public class MLSystemManager {
	
	/**
	 *  When you make a new learning algorithm, you should add a line for it to this method.
	 */
	public SupervisedLearner getLearner(String model, Random rand) throws Exception
	{
		if (model.equals("baseline")) return new BaselineLearner();
		else if (model.equals("perceptron")) return new PerceptronLearner();
		else if (model.equals("multiOutputPerceptron")) return new PerceptronMultiOutputLearner();
		else if (model.equals("backprop")) return new BackpropagationLearner();
		// else if (model.equals("neuralnet")) return new NeuralNet(rand);
		// else if (model.equals("decisiontree")) return new DecisionTree();
		// else if (model.equals("knn")) return new InstanceBasedLearner();
		else throw new Exception("Unrecognized model: " + model);
	}

	public void run(String[] args) throws Exception {

		//args = new String[]{"-L", "baseline", "-A", "data/iris.arff", "-E", "cross", "10", "-N"};

		//Random rand = new Random(1234); // Use a seed for deterministic results (makes debugging easier)
		Random rand = new Random(); // No seed for non-deterministic results

		//Parse the command line arguments
		ArgParser parser = new ArgParser(args);
		String fileName = parser.getARFF(); //File specified by the user
		String learnerName = parser.getLearner(); //Learning algorithm specified by the user
		String evalMethod = parser.getEvaluation(); //Evaluation method specified by the user
		String evalParameter = parser.getEvalParameter(); //Evaluation parameters specified by the user
		boolean printConfusionMatrix = parser.getVerbose();
		boolean normalize = parser.getNormalize();

		// Load the model
		SupervisedLearner learner = getLearner(learnerName, rand);

		// Load the ARFF file
		Matrix data = new Matrix();
		data.loadArff(fileName);
		if (normalize)
		{
			System.out.println("Using normalized data\n");
			data.normalize();
		}

		// Print some stats
		System.out.println();
		System.out.println("Dataset name: " + fileName);
		System.out.println("Number of instances: " + data.rows());
		System.out.println("Number of attributes: " + data.cols());
		System.out.println("Learning algorithm: " + learnerName);
		System.out.println("Evaluation method: " + evalMethod);
		System.out.println();

		if (evalMethod.equals("training"))
		{
			System.out.println("Calculating accuracy on training set...");
			Matrix features = new Matrix(data, 0, 0, data.rows(), data.cols() - 1);
			Matrix labels = new Matrix(data, 0, data.cols() - 1, data.rows(), 1);
			Matrix confusion = new Matrix();
			double startTime = System.currentTimeMillis();
			learner.train(features, labels);
			double elapsedTime = System.currentTimeMillis() - startTime;
			System.out.println("Time to train (in seconds): " + elapsedTime / 1000.0);
			double accuracy = learner.measureAccuracy(features, labels, confusion);
			System.out.println("Training set accuracy: " + accuracy);
			if(printConfusionMatrix) {
				System.out.println("\nConfusion matrix: (Row=target value, Col=predicted value)");
				confusion.print();
				System.out.println("\n");
			}
		}
		else if (evalMethod.equals("static"))
		{
			Matrix testData = new Matrix();
			testData.loadArff(evalParameter);
			if (normalize)
				testData.normalize(); // BUG! This may normalize differently from the training data. It should use the same ranges for normalization!

			System.out.println("Calculating accuracy on separate test set...");
			System.out.println("Test set name: " + evalParameter);
			System.out.println("Number of test instances: " + testData.rows());
			Matrix features = new Matrix(data, 0, 0, data.rows(), data.cols() - 1);
			Matrix labels = new Matrix(data, 0, data.cols() - 1, data.rows(), 1);
			double startTime = System.currentTimeMillis();
			learner.train(features, labels);
			double elapsedTime = System.currentTimeMillis() - startTime;
			System.out.println("Time to train (in seconds): " + elapsedTime / 1000.0);
			double trainAccuracy = learner.measureAccuracy(features, labels, null);
			System.out.println("Training set accuracy: " + trainAccuracy);
			Matrix testFeatures = new Matrix(testData, 0, 0, testData.rows(), testData.cols() - 1);
			Matrix testLabels = new Matrix(testData, 0, testData.cols() - 1, testData.rows(), 1);
			Matrix confusion = new Matrix();
			double testAccuracy = learner.measureAccuracy(testFeatures, testLabels, confusion);
			System.out.println("Test set accuracy: " + testAccuracy);
			if(printConfusionMatrix) {
				System.out.println("\nConfusion matrix: (Row=target value, Col=predicted value)");
				confusion.print();
				System.out.println("\n");
			}
		}
		else if (evalMethod.equals("random"))
		{
			System.out.println("Calculating accuracy on a random hold-out set...");
			double trainPercent = Double.parseDouble(evalParameter);
			if (trainPercent < 0 || trainPercent > 1)
				throw new Exception("Percentage for random evaluation must be between 0 and 1");
			System.out.println("Percentage used for training: " + trainPercent);
			System.out.println("Percentage used for testing: " + (1 - trainPercent));
			data.shuffle(rand);
			int trainSize = (int)(trainPercent * data.rows());
			Matrix trainFeatures = new Matrix(data, 0, 0, trainSize, data.cols() - 1);
			Matrix trainLabels = new Matrix(data, 0, data.cols() - 1, trainSize, 1);
			Matrix testFeatures = new Matrix(data, trainSize, 0, data.rows() - trainSize, data.cols() - 1);
			Matrix testLabels = new Matrix(data, trainSize, data.cols() - 1, data.rows() - trainSize, 1);
			double startTime = System.currentTimeMillis();
			learner.train(trainFeatures, trainLabels);
			double elapsedTime = System.currentTimeMillis() - startTime;
			System.out.println("Time to train (in seconds): " + elapsedTime / 1000.0);
			double trainAccuracy = learner.measureAccuracy(trainFeatures, trainLabels, null);
			System.out.println("Training set accuracy: " + trainAccuracy);
			Matrix confusion = new Matrix();
			double testAccuracy = learner.measureAccuracy(testFeatures, testLabels, confusion);
			System.out.println("Test set accuracy: " + testAccuracy);
			if(printConfusionMatrix) {
				System.out.println("\nConfusion matrix: (Row=target value, Col=predicted value)");
				confusion.print();
				System.out.println("\n");
			}
		}
		else if (evalMethod.equals("cross"))
		{
			System.out.println("Calculating accuracy using cross-validation...");
			int folds = Integer.parseInt(evalParameter);
			if (folds <= 0)
				throw new Exception("Number of folds must be greater than 0");
			System.out.println("Number of folds: " + folds);
			int reps = 1;
			double sumAccuracy = 0.0;
			double elapsedTime = 0.0;
			for(int j = 0; j < reps; j++) {
				data.shuffle(rand);
				for (int i = 0; i < folds; i++) {
					int begin = i * data.rows() / folds;
					int end = (i + 1) * data.rows() / folds;
					Matrix trainFeatures = new Matrix(data, 0, 0, begin, data.cols() - 1);
					Matrix trainLabels = new Matrix(data, 0, data.cols() - 1, begin, 1);
					Matrix testFeatures = new Matrix(data, begin, 0, end - begin, data.cols() - 1);
					Matrix testLabels = new Matrix(data, begin, data.cols() - 1, end - begin, 1);
					trainFeatures.add(data, end, 0, data.rows() - end);
					trainLabels.add(data, end, data.cols() - 1, data.rows() - end);
					double startTime = System.currentTimeMillis();
					learner.train(trainFeatures, trainLabels);
					elapsedTime += System.currentTimeMillis() - startTime;
					double accuracy = learner.measureAccuracy(testFeatures, testLabels, null);
					sumAccuracy += accuracy;
					System.out.println("Rep=" + j + ", Fold=" + i + ", Accuracy=" + accuracy);
				}
			}
			elapsedTime /= (reps * folds);
			System.out.println("Average time to train (in seconds): " + elapsedTime / 1000.0);
			System.out.println("Mean accuracy=" + (sumAccuracy / (reps * folds)));
		}
	}

	/**
	 * Class for parsing out the command line arguments
	 */
	private class ArgParser {
	
		String arff;
		String learner;
		String evaluation;
		String evalExtra;
		boolean verbose;
		boolean normalize;

		//You can add more options for specific learning models if you wish
		public ArgParser(String[] argv) {
			try{
	
			 	for (int i = 0; i < argv.length; i++) { // 

			 		if (argv[i].equals("-V"))
			 		{
			 			verbose = true;
			 		}
			 		else if (argv[i].equals("-N"))
			 		{
			 			normalize = true;
			 		}
						else if (argv[i].equals("-A"))
						{
							arff = argv[++i];
						}
						else if (argv[i].equals("-L"))
						{
							learner = argv[++i];
						}
						else if (argv[i].equals("-E"))
						{
							evaluation = argv[++i];
							if (argv[i].equals("static"))
							{
								//expecting a test set name
								evalExtra = argv[++i];
							}
							else if (argv[i].equals("random"))
							{
								//expecting a double representing the percentage for testing
								//Note stratification is NOT done
								evalExtra = argv[++i];
							}
							else if (argv[i].equals("cross"))
							{
								//expecting the number of folds
								evalExtra = argv[++i];
							}
							else if (!argv[i].equals("training"))
							{
								System.out.println("Invalid Evaluation Method: " + argv[i]);
								System.exit(0);
							}
						}
						else
						{
							System.out.println("Invalid parameter: " + argv[i]);
							System.exit(0);
						}
			  	}
		 
				}
				catch (Exception e) {
					System.out.println("Usage:");
					System.out.println("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E [evaluationMethod] {[extraParamters]} [OPTIONS]\n");
					System.out.println("OPTIONS:");
					System.out.println("-V Print the confusion matrix and learner accuracy on individual class values\n");
					
					System.out.println("Possible evaluation methods are:");
					System.out.println("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E training");
					System.out.println("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E static [testARFF_File]");
					System.out.println("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E random [%_ForTraining]");
				  	System.out.println("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E cross [numOfFolds]\n");
					System.exit(0);
				}
				
				if (arff == null || learner == null || evaluation == null)
				{
					System.out.println("Usage:");
					System.out.println("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E [evaluationMethod] {[extraParamters]} [OPTIONS]\n");
					System.out.println("OPTIONS:");
					System.out.println("-V Print the confusion matrix and learner accuracy on individual class values");
					System.out.println("-N Use normalized data");
					System.out.println();
					System.out.println("Possible evaluation methods are:");
					System.out.println("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E training");
					System.out.println("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E static [testARFF_File]");
					System.out.println("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E random [%_ForTraining]");
				  	System.out.println("MLSystemManager -L [learningAlgorithm] -A [ARFF_File] -E cross [numOfFolds]\n");
					System.exit(0);
				}
			}
	 
		//The getter methods
		public String getARFF(){ return arff; }	
		public String getLearner(){ return learner; }	 
		public String getEvaluation(){ return evaluation; }	
		public String getEvalParameter() { return evalExtra; }
		public boolean getVerbose() { return verbose; } 
		public boolean getNormalize() { return normalize; }
	}

	public static void main(String[] args) throws Exception
	{
		String[] copyArgs;
		if( args[args.length-1].startsWith("code=") ){
			copyArgs = new String[args.length-1];
			for( int i=0; i< args.length-1 ; i++ )  {
				copyArgs[i]= args[i];
			}
		} else {
			copyArgs = new String[args.length];
			copyArgs = args;
		}
		MLSystemManager ml = new MLSystemManager();
		ml.run(copyArgs);
	}
}
