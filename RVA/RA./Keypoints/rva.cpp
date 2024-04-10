// (c) Realidad Virtual y Aumentada - Universidad de Cordoba - Manuel J. Marin-Jimenez

#include <iostream>
#include <opencv2/opencv.hpp>

#include "rva.h"


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
