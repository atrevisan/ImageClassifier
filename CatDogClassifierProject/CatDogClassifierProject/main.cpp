#include <iostream>
#include <string>
#include "opencv/highgui.h"
#include "opencv/cv.h"
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>

#define IMAGE_SIZE 250 // normalized W and H sizes of training data

// Color classifier parameters
#define N 1
#define CH 10
#define CS 10
#define CV 10
#define NUMBER_OF_COLOR_TRAINING_SAMPLES 4000
#define NUMBER_OF_COLOR_TESTING_SAMPLES 1000
#define NUMBER_OF_COLOR_FEATURES N * N * CH * CS * CV


// Texture classifier parameters
#define NUMBER_OF_TEXELS 1000
#define DELTA 40.0
#define NUMBER_OF_TEXTURE_TRAINING_SAMPLES 4000
#define NUMBER_OF_TEXTURE_TESTING_SAMPLES 1000
#define NUMBER_OF_TEXTURE_FEATURES NUMBER_OF_TEXELS


//#define AUTO_TRAIN_SVM
#define PREDICT

#define TRAINING_DIRECTORY "C:\\Users\\Allan\\Pictures\\train\\"
#define TEST_DIRECTORY "C:\\Users\\Allan\\Pictures\\test\\"
//#define PREDICTION_DIRECTORY "C:\\Users\\Allan\\Pictures\\predict\\"
#define PREDICTION_DIRECTORY "C:\\Users\\Allan\\Pictures\\predict2\\"

#define NUMBER_OF_IMAGES_TO_PREDICT 10

using namespace cv; // Namespace do OpenCV.
using namespace std;

// for texture features extraction
vector<Mat> texels;

int totalExtracted = 0;

cv::Mat createOne(std::vector<cv::Mat> & images, int cols, int min_gap_size)
{
    // let's first find out the maximum dimensions
    int max_width = 0;
    int max_height = 0;
    for ( int i = 0; i < images.size(); i++) {
        // check if type is correct 
        // you could actually remove that check and convert the image 
        // in question to a specific type
        if ( i > 0 && images[i].type() != images[i-1].type() ) {
            std::cerr << "WARNING:createOne failed, different types of images";
            return cv::Mat();
        }
        max_height = std::max(max_height, images[i].rows);
        max_width = std::max(max_width, images[i].cols);
    }
    // number of images in y direction
    int rows = std::ceil((float) images.size() / cols);

    // create our result-matrix
    cv::Mat result = cv::Mat::zeros(rows*max_height + (rows-1)*min_gap_size,
                                    cols*max_width + (cols-1)*min_gap_size, images[0].type());
    size_t i = 0;
    int current_height = 0;
    int current_width = 0;
    for ( int y = 0; y < rows; y++ ) {
        for ( int x = 0; x < cols; x++ ) {
            if ( i >= images.size() ) // shouldn't happen, but let's be safe
                return result;
            // get the ROI in our result-image
            cv::Mat to(result,
                       cv::Range(current_height, current_height + images[i].rows),
                       cv::Range(current_width, current_width + images[i].cols));
            // copy the current image to the ROI
            images[i++].copyTo(to);
            current_width += max_width + min_gap_size;
        }
        // next line - reset width and update height
        current_width = 0;
        current_height += max_height + min_gap_size;
    }
    return result;
}

Mat resizeImage (Mat& src, int w, int h) {

	Mat dst;
	Size size(w, h);//the dst image size,e.g.100x100
	
	resize(src, dst, size);//resize image

	return dst;
}

// accuracy
void evaluateSVM(cv::Mat& testData, cv::Mat& testClasses, char* filename) {

	CvSVM svm;
	svm.load(filename);
	
	cv::Mat predicted(testClasses.rows, 1, CV_32F);
	for (int i = 0; i < testData.rows; i++) {
		const cv::Mat sample = testData.row(i);
		predicted.at<float> (i, 0) = svm.predict(sample);
	}

	assert(predicted.rows == testData.rows);
	int t = 0;
	int f = 0;

	int tp = 0;
	int fn = 0;

	int tn = 0;
	int fp = 0;

	for(int i = 0; i < testData.rows; i++) {
		float p = predicted.at<float>(i,0);
		float a = testClasses.at<float>(i,0);

		if((p >= 0.0 && a >= 0.0) || (p <= 0.0 &&  a <= 0.0)) {
			t++;
		} else {
			f++;
		}

		if (p == 1.0 && a == 1.0)
			tp++;
		else if (p == 1.0 && a == -1.0)
			fp++;
		else if (p == -1.0 && a == -1.0)
			tn++;
		else if (p == -1.0 && a == 1.0)
			fn++;
	}

	float accuracy = (t * 1.0) / (t + f);

	float positivePrecision = (tp * 1.0) / (tp + fp);
	float positiveRecall = (tp * 1.0) / (tp + fn);

	float negativePrecision = (tn * 1.0) / (tn + fn);
	float negativeRecall = (tn * 1.0) / (tn + fp);

	float positiveFMeasure = 2 * ((positivePrecision * positiveRecall)/(positivePrecision + positiveRecall));
	float negativeFMeasure = 2 * ((negativePrecision * negativeRecall)/(negativePrecision + negativeRecall));

	printf ("{SVM accuracy} %f\n", accuracy);
	printf ("{SVM positive class precision} %f\n", positivePrecision);
	printf ("{SVM positive class recall} %f\n", positiveRecall);
	printf ("{SVM negative class precision} %f\n", negativePrecision);
	printf ("{SVM negative class recall} %f\n", negativeRecall);
	printf ("{SVM positive class F measure} %f\n", positiveFMeasure);
	printf ("{SVM negative class F measure} %f\n", negativeFMeasure);
}

void trainSVM(Mat& trainingData, Mat& labels, char* filename, float C, float gamma) {

	cout << "Start SVM training, press any key to continue " << endl;
	system("pause");

	CvSVMParams params = CvSVMParams(
	      CvSVM::C_SVC,        // Type of SVM; using N classes here
	      CvSVM::RBF,  // Kernel type
	      0.0,				   // Param (degree) for poly kernel only
	      gamma,       // Param (gamma) for poly/rbf kernel only
	      0.0,				   // Param (coef0) for poly/sigmoid kernel only
	      C,             // SVM optimization param C
	      0,              // SVM optimization param nu (not used for N class SVM)
	      0,              // SVM optimization param p (not used for N class SVM)
	      NULL,           // class weights (or priors)
	      /*
	       * Optional weights, assigned to particular classes.
	       * They are multiplied by C and thus affect the misclassification
	       * penalty for different classes. The larger the weight, the larger
	       * the penalty on misclassification of data from the corresponding
	       * class.
	       */
	
			cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, 0.000001)); // Termination criteria for the learning algorithm
	
	CvSVM* svm = new CvSVM;
	//printf("Starting SVM training...\n");

	#ifdef AUTO_TRAIN_SVM
	      printf("Finding optimal parameters to use...\n");
	
	      // Use auto-training parameter grid search (ignore params manually
	      // specified above)
	      svm->train_auto(trainingData, labels, Mat(), Mat(), params, 5);
	      params = svm->get_params();
	
	      printf("The optimal parameters are: degree = %f, gamma = %f, coef0 = %f, C = %f, nu = %f, p = %f\n",
	        params.degree, params.gamma, params.coef0, params.C, params.nu, params.p);
	#else
	      printf("Using default parameters...\n");
	
	      // Use regular training with parameters manually specified above
	      svm->train(trainingData, labels, Mat(), Mat(), params);
	#endif

	printf("Training complete.\n");
	printf("Number of support vectors in the SVM: %i\n",
	svm->get_support_vector_count());

	svm -> save(filename);
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

float calculateMaxEuclideanDistance(Mat block, Mat texel) {

	int rows = block.rows;
    int cols = block.cols;

	if (rows != texel.rows || cols != texel.cols) {

		cout << "(" << rows << ", " << cols << ")" << endl; 
	
		cout << "Diferent dimension images" << endl;
		return 0.0;
	}

	vector<float> euclideanDistances;

	for (int i = 0; i < rows; i++)
    {
        Vec3b *ptr1 = block.ptr<Vec3b>(i);
		Vec3b *ptr2 = texel.ptr<Vec3b>(i);

        for (int j = 0; j < cols; j++)
        {
            Vec3b bgrPixel1 = ptr1[j];

			int B1 = bgrPixel1.val[0];
			int G1 = bgrPixel1.val[1];
			int R1 = bgrPixel1.val[2];

			Vec3b bgrPixel2 = ptr2[j];

			int B2 = bgrPixel2.val[0];
			int G2 = bgrPixel2.val[1];
			int R2 = bgrPixel2.val[2];

			float euclideanDistance = sqrt(pow((B1 - B2), 2.0) + pow((G1 - G2), 2.0) + pow((R1 - R2), 2.0));
			euclideanDistances.push_back(euclideanDistance);
        }
    }

	float maxEuclideanDistance = *max_element(euclideanDistances.begin(), euclideanDistances.end());
	return maxEuclideanDistance;
}

vector<float> normalizeEuclideanDistances(vector<float> euclideanDistances) {

	vector<float> normalizedEuclideanDistances;

	float minEuclideanDistance = *min_element(euclideanDistances.begin(), euclideanDistances.end());
	float maxEuclideanDistance = *max_element(euclideanDistances.begin(), euclideanDistances.end());

	for (int i = 0; i < euclideanDistances.size(); ++i) {
	
		float normalizedEuclideanDistance = (euclideanDistances[i] - minEuclideanDistance) / (maxEuclideanDistance - minEuclideanDistance);
		normalizedEuclideanDistances.push_back(normalizedEuclideanDistance);
	}

	return normalizedEuclideanDistances;
}

float calculateAverageEuclideanDistance(Mat texel1, Mat texel2) {

	int rows = texel1.rows;
    int cols = texel1.cols;

	if (rows != texel2.rows || cols != texel2.cols) {

		cout << "(" << rows << ", " << cols << ")" << endl; 
	
		cout << "Diferent dimension images" << endl;
		return 0.0;
	}

	float acumulatedEuclideanDistances = 0;

	for (int i = 0; i < rows; i++)
    {
        Vec3b *ptr1 = texel1.ptr<Vec3b>(i);
		Vec3b *ptr2 = texel2.ptr<Vec3b>(i);

        for (int j = 0; j < cols; j++)
        {
            Vec3b bgrPixel1 = ptr1[j];

			int B1 = bgrPixel1.val[0];
			int G1 = bgrPixel1.val[1];
			int R1 = bgrPixel1.val[2];

			Vec3b bgrPixel2 = ptr2[j];

			int B2 = bgrPixel2.val[0];
			int G2 = bgrPixel2.val[1];
			int R2 = bgrPixel2.val[2];

			float euclideanDistance = sqrt(pow((B1 - B2), 2.0) + pow((G1 - G2), 2.0) + pow((R1 - R2), 2.0));
			acumulatedEuclideanDistances += euclideanDistance;
        }
    }

	float averageEuclideanDistance = acumulatedEuclideanDistances/25;
	return averageEuclideanDistance;

}

bool isAverageEuclideanDistanceAboveTresholdForAll (Mat candidateTexel, float delta) {

	for (int i = 0; i < texels.size(); ++i) {
	
		Mat texel = texels[i];

		float averageEuclideanDistance = calculateAverageEuclideanDistance(candidateTexel, texel);

		if (averageEuclideanDistance < delta)
			return false;
	}

	return true;
}

int texelsCount = 0;
void generateSetOfTexels(int texelsSetSize, float delta) {

	cout << "Start texels set generation, press any key to continue " << endl;
	system("pause");

	while (true) {
	
		stringstream ss1;

		int cat = 0 + (rand() % (int)(9999 - 0 + 1));
		ss1 << TRAINING_DIRECTORY << "cat." << cat << ".jpg";

		Mat bgrCatImg = imread (ss1.str());
		bgrCatImg = resizeImage(bgrCatImg, IMAGE_SIZE, IMAGE_SIZE);
		
		int blockSize = IMAGE_SIZE / 50;

		for (int r = 0; r < bgrCatImg.rows; r += blockSize) {
			for (int c = 0; c < bgrCatImg.cols; c += blockSize)
			{

				Mat block = bgrCatImg(Range(r, min(r + blockSize, bgrCatImg.rows)),
									  Range(c, min(c + blockSize, bgrCatImg.cols)));

				if (isAverageEuclideanDistanceAboveTresholdForAll (block, delta)) {
				
					texels.push_back(block);

					texelsCount++;
					cout << texelsCount << endl;

					if (texelsSetSize == texels.size()) return;
				}
			}
		}

		stringstream ss2;

		int dog = 0 + (rand() % (int)(9999 - 0 + 1));
		ss2 << TRAINING_DIRECTORY << "dog." << dog << ".jpg";

		Mat bgrDogImg = imread (ss2.str());

		bgrDogImg = resizeImage(bgrDogImg, IMAGE_SIZE, IMAGE_SIZE);

		for (int r = 0; r < bgrDogImg.rows; r += blockSize) {
			for (int c = 0; c < bgrDogImg.cols; c += blockSize)
			{

				Mat block = bgrDogImg(Range(r, min(r + blockSize, bgrDogImg.rows)),
								      Range(c, min(c + blockSize, bgrDogImg.cols)));

				if (isAverageEuclideanDistanceAboveTresholdForAll (block, delta)) {
				
					texels.push_back(block);

					texelsCount++;
					cout << texelsCount << endl;

					if (texelsSetSize == texels.size()) return;
				}
			}
		}
	}

}

vector<float> extractTextureFeatures(Mat& img) {

	int textureFeaturesCount =  0;

	int blockSize = IMAGE_SIZE / 50;
	vector<float> euclideanDistancesFeatures;

	Mat centralRoi = img(Range(100, 150),
						 Range(100, 150));

	//cout << "(" << centralRoi.rows << ", " << centralRoi.cols << ")" << endl;
	//system("pause");

	for (int i = 0; i < texels.size(); ++i) {
		
		Mat texel = texels[i];
	
		vector<float> maxEuclideanDistances;

		for (int r = 0; r < centralRoi.rows; r += blockSize) {
			for (int c = 0; c < centralRoi.cols; c += blockSize)
			{

				Mat block = centralRoi(Range(r, min(r + blockSize, centralRoi.rows)),
								       Range(c, min(c + blockSize, centralRoi.cols)));

				// calculate the max euclidean distance between the pixels of block and each texel
				float maxEuclideanDistance = calculateMaxEuclideanDistance(block, texel);

				maxEuclideanDistances.push_back(maxEuclideanDistance);
			}
		}

		float minEuclideanDistance = *min_element(maxEuclideanDistances.begin(), maxEuclideanDistances.end());
		euclideanDistancesFeatures.push_back(minEuclideanDistance);
		textureFeaturesCount++;
		//cout << textureFeaturesCount << endl;
	}

	vector<float> normalizedMinEuclideanDistances = normalizeEuclideanDistances(euclideanDistancesFeatures);

	return normalizedMinEuclideanDistances;
}

void loadTextureTrainingData(Mat& dataMat, Mat& labelsMat, int dogNumber, int catNumber) {

	cout << "Start loading texture training data, press any key to continue " << endl;
	system("pause");

	vector<vector<float>> allFeatures;

	for (int i = 0; i < catNumber; ++i) {
	
		labelsMat.at<float>(i, 0) = -1.0;
 
		stringstream ss;
		ss << TRAINING_DIRECTORY << "cat." << i << ".jpg";

		Mat bgrImg = imread (ss.str());
		bgrImg = resizeImage(bgrImg, IMAGE_SIZE, IMAGE_SIZE);

		vector<float> features = extractTextureFeatures(bgrImg);
		totalExtracted++;
		cout << totalExtracted << endl;
		allFeatures.push_back(features);
	}

	for (int i = catNumber; i < catNumber + dogNumber; ++i) {
	
		labelsMat.at<float>(i, 0) = 1.0;
	}

	for (int i = 0; i < dogNumber; ++i) {
 
		stringstream ss;
		ss << TRAINING_DIRECTORY << "dog." << i << ".jpg";

		Mat bgrImg = imread (ss.str());
		bgrImg = resizeImage(bgrImg, IMAGE_SIZE, IMAGE_SIZE);

		vector<float> features = extractTextureFeatures(bgrImg);
		totalExtracted++;
		cout << totalExtracted << endl;
		allFeatures.push_back(features);
	}

	dataMat = (allFeatures.size(), allFeatures[0].size(), CV_32F);
	for(int i = 0; i < allFeatures.size(); ++i)
		dataMat.row(i) = Mat(allFeatures[i]).t();

	cout << "Finish loading texture training data, press any key to continue " << endl;
	system("pause");
}

void loadTextureTestData(Mat& dataMat, Mat& labelsMat,  int dogNumber, int catNumber) {

	cout << "Start loading texture test data, press any key to continue " << endl;
	system("pause");

	vector<vector<float>> allFeatures;

	for (int i = 0; i < catNumber; ++i) {
	
		labelsMat.at<float>(i, 0) = -1.0;
	}

	for (int i = 10000; i < 10000 + catNumber; ++i) {

 
		stringstream ss;
		ss << TEST_DIRECTORY << "cat." << i << ".jpg";

		Mat bgrImg = imread (ss.str());
		bgrImg = resizeImage(bgrImg, IMAGE_SIZE, IMAGE_SIZE);

		vector<float> features = extractTextureFeatures(bgrImg);
		totalExtracted++;
		cout << totalExtracted << endl;
		allFeatures.push_back(features);
	}

	for (int i = catNumber; i < catNumber + dogNumber; ++i) {
	
		labelsMat.at<float>(i, 0) = 1.0;
	}

	for (int i = 10000; i < 10000 + dogNumber; ++i) {

 
		stringstream ss;
		ss << TEST_DIRECTORY << "dog." << i << ".jpg";

		Mat bgrImg = imread (ss.str());
		bgrImg = resizeImage(bgrImg, IMAGE_SIZE, IMAGE_SIZE);

		vector<float> features = extractTextureFeatures(bgrImg);
		totalExtracted++;
		cout << totalExtracted << endl;
		allFeatures.push_back(features);
	}

	dataMat = (allFeatures.size(), allFeatures[0].size(), CV_32F);
	for(int i = 0; i < allFeatures.size(); ++i)
		dataMat.row(i) = Mat(allFeatures[i]).t();

	cout << "Finish loading texture test data, press any key to continue " << endl;
	system("pause");
}


vector<float> extractColorFeatures(Mat& img, int n, int ch, int cs, int cv) {

	// First thing: extract color features.
	// Color features are represented in the form of a boolean vector,
	// to compute this boolean vector first we slice the image in N^2
	// blocks, each one of equal widith and heigth which size = ImageSize / N.
	// We then partitionate the HSV color space in Ch, Cs and Cv
	// bands of equal width.
	// So, each element of the boolean vector is computed as 1
	// if for each slice block combination with each band of the
	// partitioned color space there is at least one pixel in the block
	// that falls right into the range of this  partitioned color space band, 0 otherwise.
	// The size of each feature vector is equal to N^2 * Ch * Cs * Cv
	
	vector<float> features;

	vector<vector<int>> hueIntervals		   = splitColorSpace(180, ch);
	vector<vector<int>> saturationIntervals	   = splitColorSpace(256, cs);
	vector<vector<int>> valueIntervals		   = splitColorSpace(256, cv);

	int blockSize = IMAGE_SIZE / n;

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



void loadColorTrainingData(Mat& dataMat, Mat& labelsMat, int dogNumber, int catNumber) {

	cout << "Start loading color training data, press any key to continue " << endl;
	system("pause");

	vector<vector<float>> allFeatures;

	for (int i = 0; i < catNumber; ++i) {
	
		labelsMat.at<float>(i, 0) = -1.0;

		Mat hsvImg;
 
		stringstream ss;
		ss << TRAINING_DIRECTORY << "cat." << i << ".jpg";

		Mat bgrImg = imread (ss.str());
		bgrImg = resizeImage(bgrImg, IMAGE_SIZE, IMAGE_SIZE);
		cvtColor(bgrImg, hsvImg, CV_BGR2HSV);

		vector<float> features = extractColorFeatures(hsvImg, N, CH, CS, CV);

		/*
		vector<float> features1 = extractColorFeatures(hsvImg, 1, 10, 10, 10);
		vector<float> features2 = extractColorFeatures(hsvImg, 3, 10, 8, 8);
		vector<float> features3 = extractColorFeatures(hsvImg, 5, 10, 6, 6);

		vector<float> features;
		features.reserve( features1.size() + features2.size() + features3.size()); 
		
		features.insert( features.end(), features1.begin(), features1.end());
		features.insert( features.end(), features2.begin(), features2.end());
		features.insert( features.end(), features3.begin(), features3.end());*/

		totalExtracted++;
		cout << totalExtracted << endl;
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

		vector<float> features = extractColorFeatures(hsvImg, N, CH, CS, CV);

		/*
		vector<float> features1 = extractColorFeatures(hsvImg, 1, 10, 10, 10);
		vector<float> features2 = extractColorFeatures(hsvImg, 3, 10, 8, 8);
		vector<float> features3 = extractColorFeatures(hsvImg, 5, 10, 6, 6);

		vector<float> features;
		features.reserve( features1.size() + features2.size() + features3.size()); 
		
		features.insert( features.end(), features1.begin(), features1.end());
		features.insert( features.end(), features2.begin(), features2.end());
		features.insert( features.end(), features3.begin(), features3.end());*/
		
		totalExtracted++;

		cout << totalExtracted << endl;
		allFeatures.push_back(features);
	}

	dataMat = (allFeatures.size(), allFeatures[0].size(), CV_32F);
	for(int i = 0; i < allFeatures.size(); ++i)
		dataMat.row(i) = Mat(allFeatures[i]).t();

	cout << "Finish loading color training data, press any key to continue " << endl;
	system("pause");
}

void loadColorTestData(Mat& dataMat, Mat& labelsMat,  int dogNumber, int catNumber) {

	cout << "Start loading color test data, press any key to continue " << endl;
	system("pause");

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

		vector<float> features = extractColorFeatures(hsvImg, N, CH, CS, CV);

		/*
		vector<float> features1 = extractColorFeatures(hsvImg, 1, 10, 10, 10);
		vector<float> features2 = extractColorFeatures(hsvImg, 3, 10, 8, 8);
		vector<float> features3 = extractColorFeatures(hsvImg, 5, 10, 6, 6);

		vector<float> features;
		features.reserve( features1.size() + features2.size() + features3.size()); 
		
		features.insert( features.end(), features1.begin(), features1.end());
		features.insert( features.end(), features2.begin(), features2.end());
		features.insert( features.end(), features3.begin(), features3.end());*/
		
		totalExtracted++;
		cout << totalExtracted << endl;
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

		vector<float> features = extractColorFeatures(hsvImg, N, CH, CS, CV);

		/*
		vector<float> features1 = extractColorFeatures(hsvImg, 1, 10, 10, 10);
		vector<float> features2 = extractColorFeatures(hsvImg, 3, 10, 8, 8);
		vector<float> features3 = extractColorFeatures(hsvImg, 5, 10, 6, 6);

		vector<float> features;
		features.reserve( features1.size() + features2.size() + features3.size()); 
		
		features.insert( features.end(), features1.begin(), features1.end());
		features.insert( features.end(), features2.begin(), features2.end());
		features.insert( features.end(), features3.begin(), features3.end());*/

		totalExtracted++;
		cout << totalExtracted << endl;
		allFeatures.push_back(features);
	}

	dataMat = (allFeatures.size(), allFeatures[0].size(), CV_32F);
	for(int i = 0; i < allFeatures.size(); ++i)
		dataMat.row(i) = Mat(allFeatures[i]).t();

	cout << "Finish loading color test data, press any key to continue " << endl;
	system("pause");
}

void predictCategory() {

	CvSVM colorSVM;
	colorSVM.load("colorSVM.xml");

	CvSVM colorSVM2;
	colorSVM2.load("modelosvm3.xml");

	CvSVM textureSVM;
	textureSVM.load("textureSVM.xml");

	//generateSetOfTexels(NUMBER_OF_TEXELS, DELTA);
	cv::FileStorage storage("texels.yml", cv::FileStorage::READ);
	storage["texels"] >> texels;
	storage.release();

	cout << "(colorSVM1, colorSVM2, textureSVM)" << endl;

	for (int i = 1; i < NUMBER_OF_IMAGES_TO_PREDICT + 1; ++i) {
 
		stringstream ss;
		ss << PREDICTION_DIRECTORY << "img." << i << ".jpg";

		Mat bgrImg = imread (ss.str());
		bgrImg = resizeImage(bgrImg, IMAGE_SIZE, IMAGE_SIZE);

		//namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
		//imshow( "Display window", bgrImg );                   // Show our image inside it.

		// waitKey(0);   

		Mat hsvImg;
		cvtColor(bgrImg, hsvImg, CV_BGR2HSV);

		vector<float> colorFeatures = extractColorFeatures(hsvImg, N, CH, CS, CV);
		vector<float> colorFeatures2 = extractColorFeatures(hsvImg, 5, 10, 6, 6);
		vector<float> textureFeatures = extractTextureFeatures(bgrImg);

		Mat colorFeaturesMat(1, NUMBER_OF_COLOR_FEATURES, CV_32F);
		Mat colorFeaturesMat2(1, 9000, CV_32F);
		Mat textureFeaturesMat(1, NUMBER_OF_TEXTURE_FEATURES, CV_32F);

		colorFeaturesMat.row(0) = Mat(colorFeatures).t();
		colorFeaturesMat2.row(0) = Mat(colorFeatures2).t();
		textureFeaturesMat.row(0) = Mat(textureFeatures).t();

		float colorSVMPrediction = colorSVM.predict(colorFeaturesMat);
		float colorSVMPrediction2 = colorSVM2.predict(colorFeaturesMat2);
		float textureSVMPrediction = textureSVM.predict(textureFeaturesMat);

		string colorSVMPredictionResult;
		string colorSVMPredictionResult2;
		string textureSVMPredictionResult;

		if (colorSVMPrediction == -1.0)
			colorSVMPredictionResult = "Cat";
		else
			colorSVMPredictionResult = "Dog";

		if (colorSVMPrediction2 == -1.0)
			colorSVMPredictionResult2 = "Cat";
		else
			colorSVMPredictionResult2 = "Dog";

		if (textureSVMPrediction == -1.0)
			textureSVMPredictionResult = "Cat";
		else
			textureSVMPredictionResult = "Dog";

		cout << i << "-" << "(" << colorSVMPredictionResult << ", " << colorSVMPredictionResult2 << ", " << textureSVMPredictionResult << ")" << endl;

		/*
		float combinedPrediction = (1.0/3) * colorSVMPrediction + (2.0/3) * textureSVMPrediction;

		if (combinedPrediction == -1.0) {
			
			cout << "Cat" << endl;
			//namedWindow( "Cat", WINDOW_AUTOSIZE );// Create a window for display.
			//imshow( "Cat", bgrImg);
		
		} else if (combinedPrediction == 1.0) {
		
			cout << "Dog" << endl;
			//namedWindow( "Dog", WINDOW_AUTOSIZE );// Create a window for display.
			//imshow( "Dog", bgrImg);
		}*/
		//system("pause");
	}

	vector<Mat> imgs;

	for (int i = 1; i < NUMBER_OF_IMAGES_TO_PREDICT + 1; ++i) {
 
		stringstream ss;
		ss << PREDICTION_DIRECTORY << "img." << i << ".jpg";

		Mat bgrImg = imread (ss.str());
		bgrImg = resizeImage(bgrImg, 150, 150);
		imgs.push_back(bgrImg);
	}

	Mat allInOne = createOne(imgs, 4, 3);

	namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
	imshow( "Display window", allInOne );                   // Show our image inside it.
	waitKey(0);
}


int main () {
	
#ifdef PREDICT

	predictCategory();
	
	Mat colorTestData (2000, 9000, CV_32F);
	Mat colorTestLabels (2000, 1, CV_32FC1);


	cv::FileStorage storage3("testData.yml", cv::FileStorage::READ);
	storage3["testData"] >> colorTestData;
	storage3.release();

	cv::FileStorage storage4("testLabels.yml", cv::FileStorage::READ);
	storage4["testLabels"] >> colorTestLabels;
	storage4.release();
	

	cout << "Evaluating color SVM f(5, 10, 6, 6)" << endl;
	system("pause");

	evaluateSVM(colorTestData, colorTestLabels, "colorSVM2.xml");

#else
	// SVM Color classifier
	Mat colorTrainingData(NUMBER_OF_COLOR_TRAINING_SAMPLES, NUMBER_OF_COLOR_FEATURES, CV_32F);
	Mat colorTrainingLabels(NUMBER_OF_COLOR_TRAINING_SAMPLES, 1, CV_32FC1);
	
	Mat colorTestData (NUMBER_OF_COLOR_TESTING_SAMPLES, NUMBER_OF_COLOR_FEATURES, CV_32F);
	Mat colorTestLabels (NUMBER_OF_COLOR_TESTING_SAMPLES, 1, CV_32FC1);

	// SVM Texture classifier
	Mat textureTrainingData(NUMBER_OF_TEXTURE_TRAINING_SAMPLES, NUMBER_OF_TEXTURE_FEATURES, CV_32F);
	Mat textureTrainingLabels(NUMBER_OF_TEXTURE_TRAINING_SAMPLES, 1, CV_32FC1);
	
	Mat textureTestData (NUMBER_OF_TEXTURE_TESTING_SAMPLES, NUMBER_OF_TEXTURE_FEATURES, CV_32F);
	Mat textureTestLabels (NUMBER_OF_TEXTURE_TESTING_SAMPLES, 1, CV_32FC1);

	loadColorTrainingData(colorTrainingData, colorTrainingLabels, NUMBER_OF_COLOR_TRAINING_SAMPLES/2, NUMBER_OF_COLOR_TRAINING_SAMPLES/2);
	loadColorTestData(colorTestData, colorTestLabels, NUMBER_OF_COLOR_TESTING_SAMPLES/2, NUMBER_OF_COLOR_TESTING_SAMPLES/2);

	// number of texels, delta
	generateSetOfTexels(NUMBER_OF_TEXELS, DELTA);

	cv::FileStorage storage("texels.yml", cv::FileStorage::WRITE);
	storage << "texels" << texels;
	storage.release();  

	loadTextureTrainingData(textureTrainingData, textureTrainingLabels, NUMBER_OF_TEXTURE_TRAINING_SAMPLES/2, NUMBER_OF_TEXTURE_TRAINING_SAMPLES/2);
	loadTextureTestData(textureTestData, textureTestLabels, NUMBER_OF_TEXTURE_TESTING_SAMPLES/2, NUMBER_OF_TEXTURE_TESTING_SAMPLES/2);

	trainSVM(colorTrainingData, colorTrainingLabels, "colorSVM.xml", 10, pow(10.0, -3));
	evaluateSVM(colorTestData, colorTestLabels, "colorSVM.xml");

	trainSVM(textureTrainingData, textureTrainingLabels, "textureSVM.xml", 10, pow(10.0, -1));
	evaluateSVM(textureTestData, textureTestLabels, "textureSVM.xml");
#endif

	waitKey();

	return 0;
}