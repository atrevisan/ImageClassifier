#include <iostream>
#include <string>
#include "opencv/highgui.h"
#include "opencv/cv.h"
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>


#define IMAGE_SIZE 250 // W and H sizes of training data

//#define TRAINING_DIRECTORY "C:\\Users\\Allan\\Pictures\\train\\"
#define TRAINING_DIRECTORY "C:\\Users\\Allan\\Documents\\Visual Studio 2010\\Projects\\CatDogClassifierProject\\CatDogClassifierProject\\train\\"
//#define TEST_DIRECTORY "C:\\Users\\Allan\\Pictures\\test\\"
#define TEST_DIRECTORY "C:\\Users\\Allan\\Documents\\Visual Studio 2010\\Projects\\CatDogClassifierProject\\CatDogClassifierProject\\test\\"

using namespace cv; // Namespace do OpenCV.
using namespace std;

Mat resizeImage (Mat& src, int w, int h) {

	Mat dst;
	Size size(w, h);//the dst image size,e.g.100x100
	
	resize(src, dst, size);//resize image

	return dst;
}

// accuracy
float evaluate(cv::Mat& predicted, cv::Mat& actual) {
	assert(predicted.rows == actual.rows);
	int t = 0;
	int f = 0;
	for(int i = 0; i < actual.rows; i++) {
		float p = predicted.at<float>(i,0);
		float a = actual.at<float>(i,0);
		if((p >= 0.0 && a >= 0.0) || (p <= 0.0 &&  a <= 0.0)) {
			t++;
		} else {
			f++;
		}
	}
	return (t * 1.0) / (t + f);
}

void svm(Mat& trainingData, Mat& trainingClasses, Mat& testData, Mat& testClasses) {
	CvSVMParams param = CvSVMParams();

	param.svm_type = CvSVM::C_SVC;
	param.kernel_type = CvSVM::RBF; //CvSVM::RBF, CvSVM::LINEAR ...
	param.degree = 0; // for poly
	param.gamma = 20; // for poly/rbf/sigmoid
	param.coef0 = 0; // for poly/sigmoid

	param.C = 7; // for CV_SVM_C_SVC, CV_SVM_EPS_SVR and CV_SVM_NU_SVR
	param.nu = 0.0; // for CV_SVM_NU_SVC, CV_SVM_ONE_CLASS, and CV_SVM_NU_SVR
	param.p = 0.0; // for CV_SVM_EPS_SVR

	param.class_weights = NULL; // for CV_SVM_C_SVC
	param.term_crit.type = CV_TERMCRIT_ITER +CV_TERMCRIT_EPS;
	param.term_crit.max_iter = 1000;
	param.term_crit.epsilon = 1e-6;

	// SVM training (use train auto for OpenCV>=2.0)
	CvSVM svm(trainingData, trainingClasses, Mat(), Mat(), param);

	Mat predicted(testClasses.rows, 1, CV_32F);

	for(int i = 0; i < testData.rows; i++) {
		Mat sample = testData.row(i);

		predicted.at<float>(i, 0) = svm.predict(sample);
	}

	cout << "Accuracy_{SVM} = " << evaluate(predicted, testClasses) << endl;
}

void bayes(cv::Mat& trainingData, cv::Mat& trainingClasses, cv::Mat& testData, cv::Mat& testClasses) {

	CvNormalBayesClassifier bayes(trainingData, trainingClasses);
	cv::Mat predicted(testClasses.rows, 1, CV_32F);
	for (int i = 0; i < testData.rows; i++) {
		const cv::Mat sample = testData.row(i);
		predicted.at<float> (i, 0) = bayes.predict(sample);
	}

	cout << "Accuracy_{BAYES} = " << evaluate(predicted, testClasses) << endl;
}


/* returns True if there is one or more pixels in the block in the specified intervals*/
float isThereOneOrMorePixels(Mat block, vector<int> hueInterval, vector<int> saturationInterval, 
	                                    vector<int> valueInterval) {

	int rows = block.rows;
    int cols = block.cols;

    if (block.isContinuous())
    {
        cols = rows * cols; // Loop over all pixels as 1D array.
        rows = 1;
    }

	for (int i = 0; i < rows; i++)
    {
        Vec3b *ptr = block.ptr<Vec3b>(i);
        for (int j = 0; j < cols; j++)
        {
            Vec3b hsvPixel = ptr[j];

			int H = hsvPixel.val[0];
			int S = hsvPixel.val[1];
			int V = hsvPixel.val[2];

			int minHueValue = hueInterval.at(0);
			int maxHueValue = hueInterval.at(1);

			int minSaturationValue = saturationInterval.at(0);
			int maxSaturationValue = saturationInterval.at(1);

			int minValueValue = valueInterval.at(0);
			int maxValueValue = valueInterval.at(1);
			
			if ((H >= minHueValue && H <= maxHueValue) &&
				(S >= minSaturationValue && S <= maxSaturationValue) &&
				(V >= minValueValue && V <= maxValueValue)) {
			
					return 1.0;
			}
        }
    }

	return 0.0;
}

double round(double d)
{
  return floor(d + 0.5);
}

vector<vector<int>> splitColorSpace (int size, int p) {

	vector<int> chanelValues;
	vector<vector<int>> chanelIntervals;

	for (int i = 0; i < size; ++i) {
	
		chanelValues.push_back(i);
	}

	int right = round((double)size/(double)p);
    int left = size % p;
    for(int i = 0; i < size; i+= right)
    {
        if(i < size - right)
        {
            vector<int> v;
			vector<int> vRange;
            //MAJOR CORRECTION
            for(int j = i; j < (i+right); j++)
            {
                v.push_back(chanelValues[j]);
            }
			vRange.push_back(v.front());
			vRange.push_back(v.back());
			chanelIntervals.push_back(vRange);

         //   cout << v.size() << endl;

        }
        else
        {
            right = size - i;
            vector<int> v;
			vector<int> vRange;
            //Major Correction
            for(int k =i; k < size; k++)
            {
                v.push_back(chanelValues[k]);
            }
			vRange.push_back(v.front());
			vRange.push_back(v.back());
			chanelIntervals.push_back(vRange);
           
		//	cout << v.size() << endl;

        }
	}

	return chanelIntervals;
}

vector<float> extractColorFeatures(Mat& img, int N, int Ch, int Cs, int Cv) {

	// First thing: extract color features.
	// Color features are represented in the form of a boolean vector,
	// to compute this boolean vector first we slice the image in N^2
	// blocks, each one of equal widith and heigth whitch size = ImageSize / N.
	// We then partitionate the HSV color space in Ch, Cs and Cv
	// bands of equal width.
	// So, each element of the boolean vector is computed as 1
	// if for each slice block combination with each band of the
	// partitioned color space there is at least one pixel in the block
	// that falls right into the range of this  partitioned color space band, 0 otherwise.
	// So, the size of each feature vector is equal to N^2 * Ch * Cs * Cv
	
	vector<float> features;

	vector<vector<int>> hueIntervals		   = splitColorSpace(180, Ch);
	vector<vector<int>> saturationIntervals	   = splitColorSpace(256, Cs);
	vector<vector<int>> valueIntervals		   = splitColorSpace(256, Cv);

	int blockSize = IMAGE_SIZE / N;

	for (int r = 0; r < img.rows; r += blockSize) {
		for (int c = 0; c < img.cols; c += blockSize)
		{

			Mat block = img(Range(r, min(r + blockSize, img.rows)),
							Range(c, min(c + blockSize, img.cols)));
			

			for (int x = 0; x < hueIntervals.size(); ++x) {
		
				for (int y = 0; y < saturationIntervals.size(); ++y) {
		
					for (int z = 0; z < valueIntervals.size(); ++z) {
		
						vector<int> hueInterval = hueIntervals.at(x);
						vector<int> saturationInterval = saturationIntervals.at(y);
						vector<int> valueInterval = valueIntervals.at(z);

						float feature = isThereOneOrMorePixels(block, hueInterval, saturationInterval, valueInterval);
						//cout << feature << endl;
						features.push_back(feature);
					}
				}
			}
		}
	}
	return features;
}

void loadTrainingData(Mat& dataMat, Mat& labelsMat, int dogNumber, int catNumber, int N, int Ch, int Cs, int Cv) {

	vector<vector<float>> allFeatures;

	for (int i = 0; i < catNumber; ++i) {
	
		labelsMat.at<float>(i, 0) = -1.0;

		Mat hsvImg;
 
		stringstream ss;
		ss << TRAINING_DIRECTORY << "cat." << i << ".jpg";

		Mat bgrImg = imread (ss.str());
		bgrImg = resizeImage(bgrImg, IMAGE_SIZE, IMAGE_SIZE);
		cvtColor(bgrImg, hsvImg, CV_BGR2HSV);

		vector<float> features = extractColorFeatures(hsvImg, N, Ch, Cs, Cv);
		allFeatures.push_back(features);
	}

	for (int i = catNumber; i < catNumber + dogNumber; ++i) {
	
		labelsMat.at<float>(i, 0) = 1.0;
	}

	for (int i = 0; i < dogNumber; ++i) {

		Mat hsvImg;
 
		stringstream ss;
		ss << TRAINING_DIRECTORY << "dog." << i << ".jpg";

		Mat bgrImg = imread (ss.str());
		bgrImg = resizeImage(bgrImg, IMAGE_SIZE, IMAGE_SIZE);
		cvtColor(bgrImg, hsvImg, CV_BGR2HSV);

		vector<float> features = extractColorFeatures(hsvImg, N, Ch, Cs, Cv);
		allFeatures.push_back(features);
	}

	dataMat = (allFeatures.size(), allFeatures[0].size(), CV_32F);
	for(int i = 0; i < allFeatures.size(); ++i)
		dataMat.row(i) = Mat(allFeatures[i]).t();
}

void loadTestData(Mat& dataMat, Mat& labelsMat,  int dogNumber, int catNumber, int N, int Ch, int Cs, int Cv) {

	vector<vector<float>> allFeatures;

	for (int i = 0; i < catNumber; ++i) {
	
		labelsMat.at<float>(i, 0) = -1.0;
	}

	for (int i = 10000; i < 10000 + catNumber; ++i) {

		Mat hsvImg;
 
		stringstream ss;
		ss << TEST_DIRECTORY << "cat." << i << ".jpg";

		Mat bgrImg = imread (ss.str());
		bgrImg = resizeImage(bgrImg, IMAGE_SIZE, IMAGE_SIZE);
		cvtColor(bgrImg, hsvImg, CV_BGR2HSV);

		vector<float> features = extractColorFeatures(hsvImg, N, Ch, Cs, Cv);
		allFeatures.push_back(features);
	}

	for (int i = catNumber; i < catNumber + dogNumber; ++i) {
	
		labelsMat.at<float>(i, 0) = 1.0;
	}

	for (int i = 10000; i < 10000 + dogNumber; ++i) {

		Mat hsvImg;
 
		stringstream ss;
		ss << TEST_DIRECTORY << "dog." << i << ".jpg";

		Mat bgrImg = imread (ss.str());
		bgrImg = resizeImage(bgrImg, IMAGE_SIZE, IMAGE_SIZE);
		cvtColor(bgrImg, hsvImg, CV_BGR2HSV);

		vector<float> features = extractColorFeatures(hsvImg, N, Ch, Cs, Cv);
		allFeatures.push_back(features);
	}

	dataMat = (allFeatures.size(), allFeatures[0].size(), CV_32F);
	for(int i = 0; i < allFeatures.size(); ++i)
		dataMat.row(i) = Mat(allFeatures[i]).t();
}

int main () {

	// ps: a extração das features baseadas em cores está lenta

	// 10 exemplo de treinamento, cada um com 9000 features
	Mat trainingData (10, 9000, CV_32F);
	Mat trainingLabels (10, 1, CV_32FC1);

	// 5 imagens de cachorro e 5 de gatos
	loadTrainingData (trainingData, trainingLabels, 5, 5, 5, 10, 6, 6);

	// 4 exemplos de teste
	Mat testData (4, 9000, CV_32F);
	Mat testLabels (4, 1, CV_32FC1);

	// 2 de cada
	loadTestData (testData, testLabels, 2, 2, 5, 10, 6, 6);

	// Ainda falta ajustar direito os parametros do svm
	svm(trainingData, trainingLabels, testData, testLabels);


	//bayes(trainingData, trainingLabels, testData, testLabels);

	waitKey();

	return 0;
}

