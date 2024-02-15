# MNIST 

## About 
The MNIST database is a large database of handwritten digits that is commonly used as a benchmark for image recognition algorithms, and here it is nothing more than a fun test of my ML programming ability. The algorithm simply takes in a 28x28 pixel image of a handwritten digit in `num.png`, or an image of size 28x28n in `num.png` for some integer n, and outputs the n digits it thinks the image represents. The model currently saved in `model.pt` has an accuracy of 98.74% on the testing dataset, and I challenge you to change the model to beat it!

## Usage
First, ensure all requirements are installed by running `pip install -r requirements.txt`. Then, run `python3 mnist.py` after placing `num.png` in the same directory as the script. The output will be the n digits the algorithm thinks the image represents in the terminal. 

## Example Digits
![Example Digit](num.png)
