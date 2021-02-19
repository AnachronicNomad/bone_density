# ROADMAP.md

## Process Flow

1. Acquire Image
	* Isolate area of image with "area of interest" (the bone structure)
	* Contour Mapping/Convex Hull
	* or; some sort of cropping & slicing
2. Apply image transforms
	* Preserve neighborhood mapping of pixels making up the bone structure
3. Convert transformed image to bitmap representation in defined geometry
4. Use Contouring & Gradients to identify the stratified mineral areas vs "regular"
	* May need "calibration" input from user
5. Apply bitmask for regions to "label", classify regions within the bone structure
	* May need filters or convolution in pre-processing
	* Convex hull (if accurate) can provide some of the labelling
		- Other approaches need review: it's image classification after all. 
6. Do Monte-Carlo sampling to identify proportions
	* Permutation tests?
	* Bootstrap Resampling for Confidence Interval?