#include <iostream>
#include <iomanip>
#include <time.h>
#include <opencv2/opencv.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"


using namespace std;
using namespace cv;

Mat stitch_images(Mat& img_object, Mat& img_scene, int N, int sample_size, double t, double size_factor);
Vector<int> get_uniqe_randoms_in_range(int min, int max, int n, map<string, int>& map);
Mat my_warp(const Mat& obj, const Mat& scene, const Mat& homography);

int main(int argc, const char * argv[])
{
    srand((unsigned)time(NULL));
    
    Mat img_1 = imread("D:/Dropbox/7. Semester/VIS/mini project 3/img5.jpg", CV_LOAD_IMAGE_COLOR);
    Mat img_2 = imread("D:/Dropbox/7. Semester/VIS/mini project 3/img6.jpg", CV_LOAD_IMAGE_COLOR);
    Mat img_3 = imread("D:/Dropbox/7. Semester/VIS/mini project 3/img4.jpg", CV_LOAD_IMAGE_COLOR);
    
    //imshow("Image 1", img_1);
    //imshow("Image 2", img_2);
    //imshow("Image 3", img_3);
    
    Mat stitched = stitch_images(img_2, img_1, 1000, 4, 3.0, 2.1);
    imshow("Stitched image", stitched);
    
    //Mat stitched2 = stitch_images(img_3,stitched, 10000, 4, 3.0, 2.5);
    Mat stitched2 = stitch_images(img_3, img_1, 10000, 4, 3.0, 2.5);
    imshow("Stitched image 2", stitched2);
    
    waitKey(0);
    
    cout << "Program Done!" << endl;
    return 0;
}

Mat stitch_images(Mat& img_object, Mat& img_scene, int N, int sample_size, double t, double size_factor) {
    Mat result;
    
    // Find keypoints
    int minHessian = 400;
    SurfFeatureDetector detector( minHessian );
    vector<KeyPoint> keypoints_object, keypoints_scene;
	detector.detect( img_object, keypoints_object );
	detector.detect( img_scene, keypoints_scene );
    
    // Calculate descriptors
    SurfDescriptorExtractor extractor;
	Mat descriptors_object, descriptors_scene;
	extractor.compute( img_object, keypoints_object, descriptors_object );
	extractor.compute( img_scene, keypoints_scene, descriptors_scene );
    
    // Matching using FLANN
    FlannBasedMatcher matcher;
	std::vector< DMatch > matches;
	matcher.match( descriptors_object, descriptors_scene, matches );
    
	double max_dist = 0; double min_dist = 100;
    
    for( int i = 0; i < descriptors_object.rows; i++ ) {
        double dist = matches[i].distance;
        
        if( dist < min_dist ){
            min_dist = dist;
        }
        
        if( dist > max_dist ) {
            max_dist = dist;
        }
    }
    
    cout << "-- Max dist : " <<  max_dist << endl;
    cout << "-- Min dist : " <<  min_dist << endl;

    std::vector< DMatch > good_matches;
	int counter = 0;
    
	for( int i = 0; i < descriptors_object.rows; i++ ){
		if( matches[i].distance < size_factor * min_dist ){
			good_matches.push_back( matches[i]);
			counter++;
		}
	}

    cout << "Matches found: " << good_matches.size() << endl;
    
	Mat img_matches;
    
	drawMatches( img_object, keypoints_object, img_scene, keypoints_scene,
                good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    
    imshow("Matches", img_matches);
    
    int best_score = 0;
    double best_minimum_dist = 99999;
	Mat best_homography;
    
    std::vector<Point2f> obj; // for all
    std::vector<Point2f> scene; // for all
    
    for ( int i = 0; i < good_matches.size(); i++ )
    {
        obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
        scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
    }
    
    std::vector< DMatch > final_matches;
    map<string, int> hashmap;
    for (int n = 0; n < N; n++) {
        
        // Select the sample_size random samples
        Vector<int> random_indexes = get_uniqe_randoms_in_range(0, (int)good_matches.size(), sample_size, hashmap);
        
        // Get the points from the matches
        std::vector<Point2f> obj_samples; // for the 4 arbitrary points
        std::vector<Point2f> scene_samples; // for the 4 arbitrary points
        
        for ( int i = 0; i < random_indexes.size(); i++ ) //-- Get the keypoints from the good matches
        {
            obj_samples.push_back(obj[random_indexes[i]]);
            scene_samples.push_back(scene[random_indexes[i]]);
        }
        
        // Find the homography
        Mat current_homography = findHomography( obj_samples, scene_samples, 0 );
        
        // Project all the points from obj into scene
        vector<Point2f> projected_points;
        perspectiveTransform( obj, projected_points, current_homography);
        
        // Se how many fit with their partner
        int current_score = 0;
        for (int i = 0; i < projected_points.size(); i++ ) {
            if(fabs(scene[i].x - projected_points[i].x) < t && fabs(scene[i].y - projected_points[i].y) < t){
                current_score++;
            }
        }
        
        // Save the best homography
		if(current_score > best_score){
			best_score = current_score;
			cout << "RANSAC best score: " << best_score << endl;
			best_homography = current_homography;
			cout << "Indexes: ";
			for (int ri = 0; ri < random_indexes.size(); ri++){
				cout << random_indexes[ri] << " ";
			}
			cout << endl;
			cout << "RANSAC i: " << n << endl;
		}

    }

    result = my_warp(img_object, img_scene, best_homography);
//	result = Mat::zeros(img_object.rows + img_scene.rows, img_object.cols + img_scene.cols, img_object.type() );
//    warpPerspective(img_object, result, best_homography, Size(img_object.cols + img_scene.cols, img_object.rows));
//
//    Mat mask;
//    inRange(result, Scalar(0.0, 0.0, 0.0), Scalar(0.0, 0.0, 0.0), mask);
//    imshow("Mask1", mask);
//
//    int erosion_size = 20;
//    Mat element = getStructuringElement( MORPH_RECT,
//                                        Size( 2*erosion_size + 1, 2*erosion_size+1 ),
//                                        Point( erosion_size, erosion_size ) );
//    dilate(mask, mask, element);
//    GaussianBlur(mask, mask, Size(21, 21), 10.0, 10.0);
//    //imshow("Mask2", mask);
//
//    // Copy image 1 on the first half of full image
//    Mat half(result,cv::Rect(0,0,img_scene.cols,img_scene.rows));
//    img_scene.copyTo(half, mask); // copy image2 to image1 roi

    return result;
}

Mat my_warp(const Mat& obj, const Mat& scene, const Mat& homography){
	int offset_x, offset_y;
	Mat result = Mat::zeros(scene.rows, obj.cols + scene.cols, obj.type() );
	Mat inv_homography = homography.inv();
	invert(homography, homography);
	offset_x = (result.cols / 2 ) - (scene.cols / 2);
	offset_y = 0;
	scene.copyTo(result(Rect(offset_x, offset_y, scene.cols, scene.rows)));
	for (int i = 0; i < result.rows; i++){
		for (int k = 0; k < result.cols; k++){
//	for (int i = 100; i < 102; i++){
//		for (int k = 100; k < 102; k++){
			int x = k - offset_x;
			int y = i - offset_y;
			int z = 1;
//			int nx = x*inv_homography.at<double>(0,0) + y*inv_homography.at<double>(0,1) + z*inv_homography.at<double>(0,2);
//			int ny = x*inv_homography.at<double>(1,0) + y*inv_homography.at<double>(1,1) + z*inv_homography.at<double>(1,2);
//			int nz = x*inv_homography.at<double>(2,0) + y*inv_homography.at<double>(2,1) + z*inv_homography.at<double>(2,2);

			double nx = x*homography.at<double>(0,0) + y*homography.at<double>(0,1) + z*homography.at<double>(0,2);
			double ny = x*homography.at<double>(1,0) + y*homography.at<double>(1,1) + z*homography.at<double>(1,2);
			double nz = x*homography.at<double>(2,0) + y*homography.at<double>(2,1) + z*homography.at<double>(2,2);

			// normalize
			nx = nx / nz;
			ny = ny / nz;
			nz = nz / nz;

			// object
			if (nx < obj.cols && nx >= 0 && ny < obj.rows && ny >= 0){
				//cout << "| " << homography.at<double>(0,0) << "\t" << homography.at<double>(0,1) << "\t" << homography.at<double>(0,2) << endl;
				//cout << "| " << homography.at<double>(1,0) << "\t" << homography.at<double>(1,1) << "\t" << homography.at<double>(1,2) << endl;
				//cout << "| " << homography.at<double>(2,0) << "\t" << homography.at<double>(2,1) << "\t" << homography.at<double>(2,2) << endl;
				//cout << endl;

				//cout << "x: " << nx << " y: " << ny << " z: " << nz << endl;
				//cout << endl;
				result.at<Vec3b>(i,k) = obj.at<Vec3b>((int)ny,(int)nx);
			}
		}
	}
	return result;

}

Vector<int> get_uniqe_randoms_in_range(int min, int max, int n, map<string, int>& map) {
	bool isUnique = false;
    Vector<int> result;
    int clash = 0;
    while (!isUnique){
    	result.clear();
    	while (result.size() < n) {
    	    	//srand((unsigned)time(NULL));
    	        int t = rand() % (max - min) + min;
    	        int found = 0;
    	        for (int i = 0; i < result.size(); i++) {
    	            if (result[i] == t) {
    	                found = 1;
    	            }
    	        }

    	        if (found == 0) {
    	            result.push_back(t);
    	        }
    	    }

    	    sort(result.begin(), result.end());
    	    string key = "";
    	    for (int i = 0; i < result.size(); i++) {
    	    	key += result[i] + "|";
    		}

    	    if (map.count(key) > 0){
    	    	//cout << "CLASH" << endl;
    	    	isUnique = false;
    	    	clash++;
    	    }else{
    	    	map[key] = 1;
    	    	isUnique = true;
    	    }
    }
    if (clash>0){
    	//cout << "Clashes: " << clash << endl;
    }

    return result;
}
