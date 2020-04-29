//
// Created by jiang on 2020/4/29.
//
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

const cv::Mat K = ( cv::Mat_<double> ( 3,3 ) << 931.73, 0.0, 480.0, 0.0, 933.16, 302.0, 0.0, 0.0, 1.0 );
const cv::Mat D = ( cv::Mat_<double> ( 5,1 ) << -1.7165e-1, 1.968259e-1, 0.0, 0.0, -3.639514e-1 );

void UndistortBbox(cv::Rect &rect, cv::Mat &newCamMatrix);

int main()
{
    const string str = "/home/jiang/4_learn/WeChatCode/ImageUndistort/data/";
    const int nImage = 5;
    const int ImgWidth = 960;
    const int ImgHeight = 640;

    cv::Mat map1, map2;
    cv::Size imageSize(ImgWidth, ImgHeight);
    const double alpha = 1;
    cv::Mat NewCameraMatrix = getOptimalNewCameraMatrix(K, D, imageSize, alpha, imageSize, 0);
    initUndistortRectifyMap(K, D, cv::Mat(), NewCameraMatrix, imageSize, CV_16SC2, map1, map2);

    cv::Rect Bbox{338, 141, 23, 57};

    for(int i=0; i<nImage; i++)
    {
        string InputPath = str + to_string(i) + ".png";
        cv::Mat RawImage = cv::imread(InputPath);

        cv::Mat UndistortImage;
        cv::remap(RawImage, UndistortImage, map1, map2, cv::INTER_LINEAR);
//        cv::undistort(RawImage, UndistortImage, K, D, K);

        cv::rectangle(RawImage, Bbox, cv::Scalar(255, 0, 0), 2, 1);
        cv::imshow("RawImage", RawImage);
        string OutputPath1 = str + to_string(i) + "_Bbox" + ".png";
        cv::imwrite(OutputPath1, RawImage);

        UndistortBbox(Bbox, NewCameraMatrix);
        cv::rectangle(UndistortImage, Bbox, cv::Scalar(0, 0, 255), 2, 1);
        cv::imshow("UndistortImage", UndistortImage);

        string OutputPath2 = str + to_string(i) + "_Bbox_un" + ".png";
        cv::imwrite(OutputPath2, UndistortImage);
        cv::waitKey(0);
    }

    return 0;
}

void UndistortBbox(cv::Rect &rect, cv::Mat &newCamMatrix)
{
    cv::Mat mat(4, 2,  CV_32F);
    mat.at<float>(0, 0) = rect.x;
    mat.at<float>(0, 1) = rect.y;

    mat.at<float>(1, 0) = rect.x + rect.width;
    mat.at<float>(1, 1) = rect.y;

    mat.at<float>(2, 0) = rect.x;
    mat.at<float>(2, 1) = rect.y + rect.height;

    mat.at<float>(3, 0) = rect.x + rect.width;
    mat.at<float>(3, 1) = rect.y + rect.height;

    mat = mat.reshape(2);  // 2通道，行列不变
    cv::undistortPoints(mat, mat, K, D, cv::Mat(), newCamMatrix);
    mat = mat.reshape(1);  // 单通道，行列不变

    double MaxX, MaxY;
    rect.x = min(mat.at<float>(0, 0), mat.at<float>(2, 0));
    MaxX   = max(mat.at<float>(1, 0), mat.at<float>(3, 0));
    rect.y = min(mat.at<float>(0, 1), mat.at<float>(1, 1));
    MaxY   = max(mat.at<float>(2, 1), mat.at<float>(3, 1));
    rect.width = MaxX - rect.x;
    rect.height = MaxY - rect.y;
}