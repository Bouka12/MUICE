// (c) Realidad Virtual y Aumentada - Universidad de Cordoba - Manuel J. Marin-Jimenez

#include <iostream>
#include <opencv2/opencv.hpp>

#include "rva.h"


// TASK 1

cv::Mat rva_compute_homography(std::vector<cv::Point2f> points_image1, std::vector<cv::Point2f> points_image2){
    // se usa findHomography
    cv:: Mat H = cv::findHomography(points_image1, points_image2);
    return H;
}

// rva_draw_contour

void rva_draw_contour(cv::Mat image, std::vector<cv::Point2f> points, cv::Scalar color, int thickness){
    cv::line(image, points[0], points[1],color, thickness);
    cv::line(image, points[1], points[2],color, thickness);
    cv::line(image, points[2], points[3],color, thickness);
    cv::line(image, points[3], points[0],color, thickness);

}

// rva_deform_image
void rva_deform_image(const cv::Mat & im_input, cv::Mat & im_output, cv::Mat homography){
    cv::warpPerspective(im_input, im_output, homography, im_output.size() );
}


// TASK 2


void rva_calculaKPsDesc(cv::Mat &img, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors)
{
    cv::Ptr<cv::ORB> detector = cv::ORB::create();
    //detector->detect(img, keypoints);
    //detector->compute(img, keypoints, descriptors);
    detector -> detectAndCompute(img, cv::Mat(), keypoints, descriptors);

    //detectAndCompute;
}

void rva_matchDesc(cv::Mat &descriptors1, cv::Mat &descriptors2, std::vector<cv::DMatch> &matches)
{
    if (descriptors1.empty() || descriptors2.empty()) {
    std::cerr << "Error: Empty descriptors" << std::endl;
    return; // Handle the error appropriately
    }
    
    cv::Ptr< cv::DescriptorMatcher > Objeto = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE );
    Objeto->match(descriptors1, descriptors2, matches);
    //cv::DescriptorMatcher::create();
    if (matches.empty()) {
    std::cerr << "Error: No matches found" << std::endl;
    return; // Handle the error appropriately
    }

}

void rva_dibujaMatches(cv::Mat &img1, cv::Mat &img2, std::vector<cv::KeyPoint> &keypoints1, std::vector<cv::KeyPoint> &keypoints2, std::vector<cv::DMatch> &matches, cv::Mat &img_matches)
{
    
    cv::drawMatches(img1, keypoints1, img2, keypoints2,matches,img_matches);
}


// TASK 3

void rva_localizaObj(const cv::Mat &img1, const cv::Mat &img2, const std::vector<cv::KeyPoint> &keypoints1, const std::vector<cv::KeyPoint> &keypoints2, const std::vector<cv::DMatch> &matches, cv::Mat &homography, std::vector<cv::Point2f> &pts_im2)
{
        // Convert keypoints to Point2f for homography calculation
    std::vector<cv::Point2f> obj;
    std::vector<cv::Point2f> scene;

    // Populate pts_im2 with scene points
    for (size_t i = 0; i < matches.size(); ++i) {
        obj.push_back(keypoints1[matches[i].queryIdx].pt);
        scene.push_back(keypoints2[matches[i].trainIdx].pt);
    }

    // Compute homography
    homography = cv::findHomography(obj, scene, cv::RANSAC);

    //-- Get the corners from the image_1 ( the object to be "detected" )
    std::vector<cv::Point2f> obj_corners(4);
    obj_corners[0] = cv::Point2f(0, 0);
    obj_corners[1] = cv::Point2f((float)img1.cols, 0);
    obj_corners[2] = cv::Point2f((float)img1.cols, (float)img1.rows);
    obj_corners[3] = cv::Point2f(0, (float)img1.rows);

    //std::vector<cv::Point2f> scene_corners(4);
    cv::perspectiveTransform(obj_corners, pts_im2, homography);

    /*
    // Convert keypoints to Point2f for homography calculation
    std::vector<cv::Point2f> obj;
    //std::vector<cv::Point2f> scene;

    for (size_t i = 0; i < matches.size(); ++i) {
        obj.push_back(keypoints1[matches[i].queryIdx].pt);
        pts_im2.push_back(keypoints2[matches[i].trainIdx].pt);
    }

    // Compute homography
    homography = cv::findHomography(obj, pts_im2, cv::RANSAC);
     //-- Get the corners from the image_1 ( the object to be "detected" )
    std::vector<cv::Point2f> obj_corners(4);
    obj_corners[0] = cv::Point2f(0, 0);
    obj_corners[1] = cv::Point2f( (float)img1.cols, 0 );
    obj_corners[2] = cv::Point2f( (float)img1.cols, (float)img1.rows );
    obj_corners[3] = cv::Point2f( 0, (float)img1.rows );
    std::vector<cv::Point2f> scene_corners(4);
 
    cv::perspectiveTransform( obj_corners, scene_corners, homography);

 
    //-- Draw lines between the corners (the mapped object in the scene - image_2 )
    cv::line( matches, scene_corners[0] + cv::Point2f((float)img1.cols, 0),
    scene_corners[1] + cv::Point2f((float)img1.cols, 0), cv::Scalar(0, 255, 0), 4 );
    cv::line( matches, scene_corners[1] + cv::Point2f((float)img1.cols, 0),
    scene_corners[2] + cv::Point2f((float)img1.cols, 0), cv::Scalar( 0, 255, 0), 4 );
    cv::line( matches, scene_corners[2] + cv::Point2f((float)img1.cols, 0),
    scene_corners[3] + cv::Point2f((float)img1.cols, 0), cv::Scalar( 0, 255, 0), 4 );
    cv::line( matches, scene_corners[3] + cv::Point2f((float)img1.cols, 0),
    scene_corners[0] + cv::Point2f((float)img1.cols, 0), cv::Scalar( 0, 255, 0), 4 );*/
}
