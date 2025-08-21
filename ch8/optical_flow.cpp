#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <string>
#include <chrono>
#include <functional>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace cv;

string file_1 = "../ch8/images/LK1.png";
string file_2 = "../ch8/images/LK2.png";


class OpticalFlowTracker 
{
public:
    OpticalFlowTracker(const Mat &img1_, const Mat &img2_,
                        const vector<KeyPoint> &kp1_,
                        vector<KeyPoint> &kp2_,
                        vector<bool> &success_,
                        bool inverse_ = true, bool has_initial_ = false)
            : img1(img1_), img2(img2_), kp1(kp1_), kp2(kp2_), success(success_),
            inverse(inverse_), has_initial(has_initial_) {}
    void calculateOpticalFlow(const Range &range);

private:
    const Mat &img1;
    const Mat &img2;
    const vector<KeyPoint> &kp1;
    vector<KeyPoint> &kp2;
    vector<bool> &success;
    bool inverse = true;
    bool has_initial = false;
};

void OpticalFlowSingleLevel(const Mat &img1, const Mat &img2,
                            const vector<KeyPoint> &kp1, vector<KeyPoint> &kp2,
                            vector<bool> &success, bool inverse = false,
                            bool has_initial_guess = false);

void OpticalFlowMultiLevel(const Mat &img1, const Mat &img2,
                            const vector<KeyPoint> &kp1, vector<KeyPoint> &kp2,
                            vector<bool> &success, bool inverse = false);

inline float GetPixelValue(const cv::Mat &img, float x, float y) 
{
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= img.cols - 1) x = img.cols - 2;
    if (y >= img.rows - 1) y = img.rows - 2;

    float xx = x - floor(x);
    float yy = y - floor(y);
    int x_a1 = std::min(img.cols - 1, int(x) + 1);
    int y_a1 = std::min(img.rows - 1, int(y) + 1);

    return (1 - xx) * (1 - yy) * img.at<uchar>(int(y), int(x))
         + xx * (1 - yy) * img.at<uchar>(int(y), x_a1)
         + (1 - xx) * yy * img.at<uchar>(y_a1, int(x))
         + xx * yy * img.at<uchar>(y_a1, x_a1);
}

int main(int argc, char **argv) 
{
    Mat img1 = imread(file_1, IMREAD_GRAYSCALE);
    Mat img2 = imread(file_2, IMREAD_GRAYSCALE);

    if (img1.empty() || img2.empty()) 
    {
        cerr << "Failed to load images: " << file_1 << " or " << file_2 << endl;
        return -1;
    }


    vector<KeyPoint> kp1;
    Ptr<GFTTDetector> detector = GFTTDetector::create(500, 0.01, 20);
    detector->detect(img1, kp1);


    vector<KeyPoint> kp2_single;
    vector<bool> success_single;
    OpticalFlowSingleLevel(img1, img2, kp1, kp2_single, success_single);


    vector<KeyPoint> kp2_multi;
    vector<bool> success_multi;
    auto t1 = chrono::steady_clock::now();
    OpticalFlowMultiLevel(img1, img2, kp1, kp2_multi, success_multi, true);
    auto t2 = chrono::steady_clock::now();
    auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "optical flow by gauss-newton: " << time_used.count() << endl;

    // OpenCV Pyramidal LK for validation
    vector<Point2f> pt1, pt2;
    pt1.reserve(kp1.size());

    for (auto &kp : kp1) 
    {
        pt1.push_back(kp.pt);
    }

    vector<uchar> status;
    vector<float> error;
    t1 = chrono::steady_clock::now();
    cv::calcOpticalFlowPyrLK(img1, img2, pt1, pt2, status, error);

    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);

    cout << "optical flow by opencv: " << time_used.count() << endl;

    // Visualization
    auto toBGR = [](const Mat& src) -> Mat 
    {
        Mat u8 = src;
        if (u8.depth() != CV_8U) 
            u8.convertTo(u8, CV_8U);
        Mat out;
        if (u8.channels() == 1) 
            cv::cvtColor(u8, out, cv::COLOR_GRAY2BGR);
        else if (u8.channels() == 3) 
            out = u8.clone();
        else if (u8.channels() == 4) 
            cv::cvtColor(u8, out, cv::COLOR_BGRA2BGR);
        else { out = Mat(); }
        
        return out;
    };

    Mat img2_single = toBGR(img2);
    Mat img2_multi  = toBGR(img2);
    Mat img2_CV     = toBGR(img2);

    CV_Assert(!img2_single.empty() && !img2_multi.empty() && !img2_CV.empty());

    int n1 = (int)min({kp2_single.size(), kp1.size(), success_single.size()});

    for (int i = 0; i < n1; i++) 
    {
        if (success_single[i]) 
        {
            cv::circle(img2_single, kp2_single[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_single, kp1[i].pt, kp2_single[i].pt, cv::Scalar(0, 250, 0));
        }
    }

    int n2 = (int)min({kp2_multi.size(), kp1.size(), success_multi.size()});

    for (int i = 0; i < n2; i++) 
    {
        if (success_multi[i]) 
        {
            cv::circle(img2_multi, kp2_multi[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_multi, kp1[i].pt, kp2_multi[i].pt, cv::Scalar(0, 250, 0));
        }
    }

    int n3 = (int)min({pt2.size(), pt1.size(), status.size()});

    for (int i = 0; i < n3; i++) 
    {
        if (status[i]) 
        {
            cv::circle(img2_CV, pt2[i], 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_CV, pt1[i], pt2[i], cv::Scalar(0, 250, 0));
        }
    }

    cv::imshow("tracked single level", img2_single);
    cv::imshow("tracked multi level", img2_multi);
    cv::imshow("tracked by opencv", img2_CV);
    cv::waitKey(0);

    return 0;
}

void OpticalFlowSingleLevel(const Mat &img1, const Mat &img2,
                            const vector<KeyPoint> &kp1, vector<KeyPoint> &kp2,
                            vector<bool> &success, bool inverse, bool has_initial) 
{
    kp2.resize(kp1.size());
    success.resize(kp1.size());
    OpticalFlowTracker tracker(img1, img2, kp1, kp2, success, inverse, has_initial);
    parallel_for_(Range(0, (int)kp1.size()),
                    std::bind(&OpticalFlowTracker::calculateOpticalFlow, &tracker, std::placeholders::_1));
}

void OpticalFlowTracker::calculateOpticalFlow(const Range &range) 
{
    int half_patch_size = 4;
    int iterations = 10;

    for (int i = range.start; i < range.end; i++) 
    {
        auto kp = kp1[i];
        double dx = 0, dy = 0;

        if (has_initial) 
        {
            dx = kp2[i].pt.x - kp.pt.x;
            dy = kp2[i].pt.y - kp.pt.y;
        }

        double cost = 0, lastCost = 0;
        bool succ = true;

        Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
        Eigen::Vector2d b = Eigen::Vector2d::Zero();
        Eigen::Vector2d J;

        for (int iter = 0; iter < iterations; iter++) 
        {
            if (!inverse) 
            {
                H.setZero();
                b.setZero();
            } 

            else 
            {
                b.setZero();
            }
            cost = 0;

            for (int x = -half_patch_size; x < half_patch_size; x++)
                for (int y = -half_patch_size; y < half_patch_size; y++) 
                {
                    double error = GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y) -
                                    GetPixelValue(img2, kp.pt.x + x + dx, kp.pt.y + y + dy);

                    if (!inverse) 
                    {
                        J = -1.0 * Eigen::Vector2d(
                            0.5 * (GetPixelValue(img2, kp.pt.x + dx + x + 1, kp.pt.y + dy + y) -
                                    GetPixelValue(img2, kp.pt.x + dx + x - 1, kp.pt.y + dy + y)),
                            0.5 * (GetPixelValue(img2, kp.pt.x + dx + x, kp.pt.y + dy + y + 1) -
                                    GetPixelValue(img2, kp.pt.x + dx + x, kp.pt.y + dy + y - 1))
                        );
                    } 

                    else if (iter == 0) 
                    {
                        J = -1.0 * Eigen::Vector2d(
                            0.5 * (GetPixelValue(img1, kp.pt.x + x + 1, kp.pt.y + y) -
                                    GetPixelValue(img1, kp.pt.x + x - 1, kp.pt.y + y)),
                            0.5 * (GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y + 1) -
                                    GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y - 1))
                        );
                    }

                    b += -error * J;
                    cost += error * error;

                    if (!inverse || iter == 0) H += J * J.transpose();
                }

            Eigen::Vector2d update = H.ldlt().solve(b);

            if (std::isnan(update[0])) { succ = false; break; }
            if (iter > 0 && cost > lastCost) break;

            dx += update[0];
            dy += update[1];
            lastCost = cost;
            succ = true;

            if (update.norm() < 1e-2) break;
        }

        success[i] = succ;
        kp2[i].pt = kp.pt + Point2f((float)dx, (float)dy);
    }
}

void OpticalFlowMultiLevel(const Mat &img1, const Mat &img2,
                            const vector<KeyPoint> &kp1, vector<KeyPoint> &kp2,
                            vector<bool> &success, bool inverse) {
    int pyramids = 4;
    double pyramid_scale = 0.5;
    double scales[] = {1.0, 0.5, 0.25, 0.125};

    auto t1 = chrono::steady_clock::now();
    vector<Mat> pyr1, pyr2;

    for (int i = 0; i < pyramids; i++) 
    {
        if (i == 0) 
        {
            pyr1.push_back(img1);
            pyr2.push_back(img2);
        } 

        else 
        {
            Mat img1_pyr, img2_pyr;
            cv::resize(pyr1[i - 1], img1_pyr,
                       cv::Size((int)(pyr1[i - 1].cols * pyramid_scale),
                                (int)(pyr1[i - 1].rows * pyramid_scale)));
            cv::resize(pyr2[i - 1], img2_pyr,
                       cv::Size((int)(pyr2[i - 1].cols * pyramid_scale),
                                (int)(pyr2[i - 1].rows * pyramid_scale)));
            pyr1.push_back(img1_pyr);
            pyr2.push_back(img2_pyr);
        }
    }
    auto t2 = chrono::steady_clock::now();
    auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "build pyramid time: " << time_used.count() << endl;

    vector<KeyPoint> kp1_pyr, kp2_pyr;
    kp1_pyr.reserve(kp1.size());
    kp2_pyr.reserve(kp1.size());

    for (auto &kp : kp1) 
    {
        auto kp_top = kp;
        kp_top.pt *= (float)scales[pyramids - 1];
        kp1_pyr.push_back(kp_top);
        kp2_pyr.push_back(kp_top);
    }

    for (int level = pyramids - 1; level >= 0; level--) 
    {
        success.clear();
        t1 = chrono::steady_clock::now();
        OpticalFlowSingleLevel(pyr1[level], pyr2[level], kp1_pyr, kp2_pyr, success, inverse, true);
        t2 = chrono::steady_clock::now();
        auto time_used_l = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
        cout << "track pyr " << level << " cost time: " << time_used_l.count() << endl;

        if (level > 0) 
        {
            for (auto &kp : kp1_pyr) kp.pt /= (float)pyramid_scale;
            for (auto &kp : kp2_pyr) kp.pt /= (float)pyramid_scale;
        }
    }

    kp2 = kp2_pyr;
}
