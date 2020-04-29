//
// Created by jiang on 2020/4/29.
//
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

const cv::Mat K = ( cv::Mat_<double> ( 3,3 ) << 931.73, 0.0, 480.0, 0.0, 933.16, 302.0, 0.0, 0.0, 1.0 );
const cv::Mat D = ( cv::Mat_<double> ( 5,1 ) << -1.7165e-1, 1.968259e-1, 0.0, 0.0, -3.639514e-1 );

void UndistortKeyPoints(vector<cv::Point2f> &points);

int main()
{
    const string str = "/home/jiang/4_learn/WeChatCode/ImageUndistort/data/";
    const int nImage = 5;
    const int MAX_CNT = 150;
    const int MIN_DIST = 30;

    for(int i=0; i<nImage; i++)
    {
        string InputPath = str + to_string(i) + ".png";
        cv::Mat RawImage = cv::imread(InputPath);

        vector<cv::Point2f> pts;
        cv::Mat RawImage_Gray;
        cv::cvtColor(RawImage, RawImage_Gray, CV_RGB2GRAY);

        cv::goodFeaturesToTrack(RawImage_Gray, pts, MAX_CNT, 0.01, MIN_DIST);

        for(auto& pt:pts)
            circle(RawImage, pt, 2, cv::Scalar(255, 0, 0), 2);
        cv::imshow("pts", RawImage);

        UndistortKeyPoints(pts);

        cv::Mat UndistortImage;
        cv::undistort(RawImage, UndistortImage, K, D, K);

        for(auto& pt:pts)
            circle(UndistortImage, pt, 2, cv::Scalar(0, 0, 255), 2);
        cv::imshow("pts_un", UndistortImage);

        string OutputPath = str + to_string(i) + "_pts_un" + ".png";
        cv::imwrite(OutputPath, UndistortImage);
        cv::waitKey(0);
    }

    return 0;
}

void UndistortKeyPoints(vector<cv::Point2f> &points)
{
    if(D.at<float>(0)==0.0)    // 图像矫正过
        return;

    // N为提取的特征点数量，将N个特征点保存在N*2的mat中
    uint N = points.size();
    cv::Mat mat(N,2,CV_32F);
    for(int i=0; i<N; i++)
    {
        mat.at<float>(i,0)=points[i].x;
        mat.at<float>(i,1)=points[i].y;
    }

    // 调整mat的通道为2，矩阵的行列形状不变
    mat=mat.reshape(2);
    cv::undistortPoints(mat, mat, K, D, cv::Mat(), K);
    mat=mat.reshape(1);

    // 存储校正后的特征点
    for(int i=0; i<N; i++)
    {
        cv::Point2f kp = points[i];
        kp.x=mat.at<float>(i,0);
        kp.y=mat.at<float>(i,1);
        points[i] = kp;
    }
}