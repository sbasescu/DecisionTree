/*
 * Simon Basescu, Charlie Ferguson
 * Group honor code: “All group members were present and contributing during all work on this project”
 * On our honor, we have neither given nor received unauthorized aid on this assignment. 
 */


import java.util.ArrayList;


public class DecisionTree {
	private TreeNode root = null; //stores the root of the decision tree
	
	
	public void train(ArrayList<Example> examples){
		int numFeatures = 0;
		if(examples.size()>0) //get the number of featuers in these examples
			numFeatures = examples.get(0).getNumFeatures();
		
		//initialize empty positive and negative lists
		ArrayList<Example> pos = new ArrayList<Example>();
		ArrayList<Example> neg = new ArrayList<Example>();
		
		//paritition examples into positive and negative ones
		for(Example e: examples){
			if (e.getLabel())
				pos.add(e);
			else
				neg.add(e);
		}

		//create the root node of the tree
		root = new TreeNode(null, pos, neg, numFeatures);
		
		//call recursive train()  on the root node
		train(root, numFeatures);
	}
	
	/**
	 * TODO: Complete this method
	 * The recursive train method that builds a tree at TreeNode node
	 * @param node: current node to train
	 * @param numFeatures: total number of features
	 */
	private void train(TreeNode node, int numFeatures){
		//Base Case 1
		if (node.pos.size() == 0 && node.neg.size() > 0) {
			node.isLeaf = true;
			// Set label to false
			node.decision = false;
		}
		else if (node.neg.size() == 0 && node.pos.size() > 0){
			node.isLeaf = true;
			// Set label to true
			node.decision = true;
		}
		//Base Case 2
		else if (node.neg.size() == 0 && node.pos.size() == 0) {
			node.decision = node.parent.pos.size() >= node.parent.neg.size() ? true : false; // use parent's values
			node.isLeaf = true;
		}
		// Base Case 3
		else if (node.getNumberFeaturesUsed() == numFeatures) { // if all features used
			node.isLeaf = true;
			node.decision = node.pos.size() >= node.neg.size() ? true : false;
		}
		
		else {
			int featureSplit = -1; // keeps track of index of feature with highest info gain
			double maxInfoGain = -1; // keeps track of the maxInfoGain
			
			for (int i = 0; i < numFeatures; i++) {
				if (!node.featureUsed(i)) {
					//info gain of feature i
					double currInfoGain = getEntropy(node.pos.size(), node.neg.size()) - getRemainingEntropy(i, node);
					
					// if info gain for feature i is greater than the current max info gain of other features, update 
					// feature to split on to be i, and update new maxInfoGain
					if (currInfoGain > maxInfoGain) {
						maxInfoGain = currInfoGain;
						featureSplit = i; 
					}
				}
			}
			node.setSplitFeature(featureSplit);
			
			createChildren(node, numFeatures);
			train(node.trueChild, numFeatures);
			train(node.falseChild, numFeatures);
		}
	}
	
	/**
	 * TODO: Complete this method
	 * Creates the true and false children of TreeNode node
	 * @param node: node at which to create children
	 * @param numFeatures: total number of features
	 */
	private void createChildren(TreeNode node, int numFeatures){
		int featureSplit = node.getSplitFeature(); 
		
		ArrayList<Example> posListTrueFeature = new ArrayList<Example>(); // In positive list, true for feature
		ArrayList<Example> posListFalseFeature = new ArrayList<Example>();// In positive list, false for feature
		
		
		ArrayList<Example> negListTrueFeature = new ArrayList<Example>(); // In negative list, true for feature
		ArrayList<Example> negListFalseFeature = new ArrayList<Example>(); // In negative list, false for feature
		

		// Tally the positive list 
		for (int i = 0; i < node.pos.size(); i++) {
			if (node.pos.get(i).getFeatureValue(featureSplit))
				posListTrueFeature.add(node.pos.get(i));
			else 
				posListFalseFeature.add(node.pos.get(i));
			
		}
		// Tally the negative list
		for (int i = 0; i < node.neg.size(); i++) {
			if (node.neg.get(i).getFeatureValue(featureSplit))
				negListTrueFeature.add(node.neg.get(i));
			else 
				negListFalseFeature.add(node.neg.get(i));
		}
		
		//create the children nodes of the tree
		TreeNode trueChild = new TreeNode(node, posListTrueFeature, negListTrueFeature, numFeatures);
		TreeNode falseChild = new TreeNode(node, posListFalseFeature, negListFalseFeature, numFeatures);
		
		node.falseChild = falseChild;
		node.trueChild = trueChild; 
	}
	
	
	/**
	 * TODO: Complete this method
	 * Computes and returns the remaining entropy if feature is chosen
	 * at node.
	 * @param feature: the feature number
	 * @param node: node at which to find remaining entropy
	 * @return remaining entropy at node
	 */
	private double getRemainingEntropy(int feature, TreeNode node){
		
		int numPosListTrueFeature = 0; // In positive list, true for feature
		int numPosListFalseFeature = 0; // In positive list, false for feature
		
		int numNegListTrueFeature = 0; // In negative list, true for feature
		int numNegListFalseFeature = 0; // In negative list, False for feature
		
		
		// Tally the positive list 
		for (int i = 0; i < node.pos.size(); i++) {
			if (node.pos.get(i).getFeatureValue(feature))
				numPosListTrueFeature++;
			else 
				numPosListFalseFeature++;
			
		}
		// Tally the negative list
		for (int i = 0; i < node.neg.size(); i++) {
			if (node.neg.get(i).getFeatureValue(feature))
				numNegListTrueFeature++;
			else 
				numNegListFalseFeature++;
		}
		
		// Calculate the remaining entropy
		double trueEntropy = getEntropy(numPosListTrueFeature, numNegListTrueFeature);
		
		double falseEntropy = getEntropy(numPosListFalseFeature, numNegListFalseFeature);
		
		double sumTrue = (double)numPosListTrueFeature + numNegListTrueFeature;
		double sumFalse = (double)numPosListFalseFeature + numNegListFalseFeature;

		double totalEx = (double) node.pos.size() + node.neg.size();
	
		double probTrue = (sumTrue / totalEx);
		double probFalse = (sumFalse / totalEx);
		
		return (double) (trueEntropy * probTrue + falseEntropy * probFalse);
	
	}
	
	/**
	 * TODO: complete this method
	 * Computes the entropy of a node given the number of positive and negative examples it has
	 * @param numPos: number of positive examples
	 * @param numNeg: number of negative examples
	 * @return - entropy
	 */
	private double getEntropy(int numPos, int numNeg){
		
		double numP = (double) (numPos);
		double numN = (double) (numNeg);
			
		double total = numP + numN;
		double fracPos = numP / total;
		double fracNeg = numN / total;
		
		// if numPos and numNeg is 0, entropy for that term is 0
		if (numPos == 0 && numNeg == 0) 
			return 0;

		// if numPos or numNeg is 0 but the other is positive, only calculate entropy for non-negative since
		// entropy of the other term will be zero
		if (numPos == 0 && numNeg > 0) {
			return (- log2(fracNeg) * fracNeg);
		} 
		else if (numNeg == 0 && numPos > 0)
			return (- log2(fracPos) * fracPos);
		
		// calculate remaining entropy for both positive and negative
		return (-log2(fracPos) * fracPos - log2(fracNeg) * fracNeg);
	}
	
	/**	
	 * Computes log_2(d) (To be used by the getEntropy() method)
	 * @param d - value
	 * @return log_2(d)
	 */
	private double log2(double d){
		return Math.log(d)/Math.log(2);
	}
	
	/** 
	 * TODO: complete this method
	 * Classifies example e using the learned decision tree
	 * @param e: example
	 * @return true if e is predicted to be  positive,  false otherwise
	 */
	public boolean classify(Example e){
		TreeNode curr = root;
		
		while(!curr.isLeaf) {
			int feature = curr.getSplitFeature();
			boolean val = e.getFeatureValue(feature);
			
			if (val)
				curr = curr.trueChild;
			else
				curr = curr.falseChild;
			
		}
		
		return curr.decision;
	}
	
	
	
	
	//----------DO NOT MODIFY CODE BELOW------------------
	public void print(){
		printTree(root, 0);
	}
	

	
	private void printTree(TreeNode node, int indent){
		if(node== null)
			return;
		if(node.isLeaf){
			if(node.decision)
				System.out.println("Positive");
			else
				System.out.println("Negative");
		}
		else{
			System.out.println();
			doIndents(indent);
			System.out.print("Feature "+node.getSplitFeature() + " = True:" );
			printTree(node.trueChild, indent+1);
			doIndents(indent);
			System.out.print("Feature "+node.getSplitFeature() + " = False:" );//+  "( " + node.falseChild.pos.size() + ", " + node.falseChild.neg.size() + ")");
			printTree(node.falseChild, indent+1);
		}
	}
	
	private void doIndents(int indent){
		for(int i=0; i<indent; i++)
			System.out.print("\t");
	}
}
