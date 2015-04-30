Files:

	This source code package includes the following files:

	Zhang_Project1.m:	main function for neural network training and validating. Call loop.m and validate.m.

	loop.m:				Training code, update weights between each two layer nodes.

	validate.m: 		Validate input sets with current weights.

	generalization.m: 	Reducing inputs from 12 by 8 to 6 by 4 and train/validate/test in the general neural network. Calling loop.m and validate.m.

	k_fold.m:			Re-divide learning and validating sets into 10 folds, and at each iteration it is trained by 9 folds and validated by 1 fold. Rotationally choose learning data and validating data folds, using general neural network to train/validate/test.

	eta_plot.m:			Plot epoch-error curves with various learning rates.

	hidden_plot.m		Plot epoch-error curves with various number of hidden units.

	sigmoid.m			Implementing sigmoid function.

	decaying.m			Decay learning rate as iteration number increases.

	best.mat			Saved best neural network. Used in testing mode.

Usage:
	
	1. There are three ways to run a matlab file example.m: 
		- Open matlab GUI and code file, then click the 'run' icon or press F5 button; 
		- Open the shell command line, run "matlab -r example" (without .m)
		- Open matlab command window without GUI display, type "run example" (without .m). 
	
	2. Run Zhang_Project1.m file in matlab GUI, or in the shell command line "matlab -r Zhang_Project1" (without .m), or in the Matlab command line "run Zhang_Project1", then it will ask which mode do you want: training or testing. Now type "training" or "testing".
	In the training mode, input the learning rate and the number of hidden units and the code will output the classification rate for each character, the confusion matrix, the overall classification rate, as well as the training time.

	In the testing mode, the best ANN has been found and saved, so you only need to specify the file to be tested, following fomrat: path/filename, e.g.: learn-grid/a1, validate-grid/c1, test-grid/a251. Please note that in the test-grid folder, all files are named from 251.

	3. Run generalization.m, it will return the successful classification rate and the training time. 

	4. Run k_fold.m, it will return the successful rate and the training time in k-fold approach.

	5. Run decaying.m, the learning rate decreases 5% in each iteration. If the network did not converge quickly, it will take a very long time to reach the converge point. Because as the iteration increasing, the learning rate decreases sharply, which might be too small to reach the training stop condition. In this case, just kill the program and try again.
