#include <iostream>
#include <vector>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include <opencv2/features2d.hpp>

using namespace std;
using namespace cv;

static string kVideoPath = "../ch8/images/slow_traffic_small.mp4";

int main()
{
    VideoCapture cap(kVideoPath);
    if (!cap.isOpened())
    {
        cerr << "Unable to open video: " << kVideoPath << endl;
        return -1;
    }

    // ORB + BF-Hamming (Feature Matching)
    Ptr<ORB> orb = ORB::create(1000);
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    // LK (Optical Flow)
    const int maxCorners = 300;
    const double qualityLevel = 0.01;
    const double minDistance = 7.0;
    const int blockSize = 7;

    Mat prevFrame, prevGray;
    if (!cap.read(prevFrame) || prevFrame.empty())
    {
        cerr << "Cannot read first frame.\n";
        return -1;
    }
    cvtColor(prevFrame, prevGray, COLOR_BGR2GRAY);

    // LK 
    vector<Point2f> pts_prev;
    goodFeaturesToTrack(prevGray, pts_prev, maxCorners, qualityLevel, minDistance, Mat(), blockSize, false, 0.04);

    // ORB 
    vector<KeyPoint> kp_prev;
    Mat desc_prev;
    orb->detectAndCompute(prevGray, noArray(), kp_prev, desc_prev);

    // LK tracking
    Mat flowMask = Mat::zeros(prevFrame.size(), prevFrame.type());

    for (;;)
    {
        Mat frame, gray;
        if (!cap.read(frame) || frame.empty())
            break;

        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // ===== (A) ORB Feature Matching (prev ↔ cur) =====
        vector<KeyPoint> kp_cur;
        Mat desc_cur;
        orb->detectAndCompute(gray, noArray(), kp_cur, desc_cur);

        Mat matchVis;

        if (!desc_prev.empty() && !desc_cur.empty())
        {
            // KNN matching + Ratio Test
            vector<vector<DMatch>> knnMatches;
            matcher->knnMatch(desc_prev, desc_cur, knnMatches, 2);
            const float ratio = 0.75f;

            vector<DMatch> good;
            good.reserve(knnMatches.size());

            for (const auto &m : knnMatches)
            {
                if (m.size() < 2) 
                    continue;
                
                if (m[0].distance < ratio * m[1].distance)
                    good.push_back(m[0]);
            }

            drawMatches(prevFrame, kp_prev, frame, kp_cur, good, matchVis,
                        Scalar::all(-1), Scalar::all(-1),
                        vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        }

        else
        {
            matchVis = frame.clone();
            putText(matchVis, "No descriptors", Point(20, 40),
                    FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2);
        }

        // ===== (B) Pyramidal LK Optical Flow (prev → cur) =====
        vector<Point2f> pts_cur;
        vector<uchar> status;
        vector<float> err;

        if (!pts_prev.empty())
        {
            TermCriteria criteria((TermCriteria::COUNT | TermCriteria::EPS), 10, 0.03);
            calcOpticalFlowPyrLK(prevGray, gray, pts_prev, pts_cur, status, err,
                                    Size(15, 15), 2, criteria);

            // 그리기용 BGR 이미지 + 누적 마스크 업데이트
            Mat flowVis = frame.clone();
            for (size_t i = 0; i < pts_prev.size(); ++i)
            {
                if (status[i])
                {
                    line(flowMask, pts_cur[i], pts_prev[i], Scalar(0, 255, 0), 2);
                    circle(flowVis, pts_cur[i], 3, Scalar(0, 0, 255), -1);
                }
            }

            add(flowVis, flowMask, flowVis);

            imshow("B) LK Optical Flow (dense-ish over selected corners)", flowVis);

            // 다음 프레임을 위해 포인트/이미지 갱신
            vector<Point2f> pts_good;
            pts_good.reserve(pts_cur.size());

            for (size_t i = 0; i < pts_cur.size(); ++i)
                if (status[i]) pts_good.push_back(pts_cur[i]);

            pts_prev = pts_good;
        }

        else
        {
            // 포인트가 다 사라지면 다시 추출
            goodFeaturesToTrack(prevGray, pts_prev, maxCorners, qualityLevel, minDistance, Mat(), blockSize, false, 0.04);
        }

        // 매칭 시각화 출력
        imshow("A) ORB Feature Matching (prev vs cur)", matchVis);

        // 다음 루프 준비
        prevGray = gray.clone();
        prevFrame = frame.clone();
        kp_prev = kp_cur;
        desc_prev = desc_cur;

        // LK용 포인트가 너무 줄면 다시 보충
        if (pts_prev.size() < 50)
        {
            vector<Point2f> replenish;
            goodFeaturesToTrack(prevGray, replenish, maxCorners, qualityLevel, minDistance, Mat(), blockSize, false, 0.04);
            // 기존 surviving 포인트에 추가 (간단히 덮어쓰기 대신 합치기)
            pts_prev.insert(pts_prev.end(), replenish.begin(), replenish.end());

            if (pts_prev.size() > (size_t)maxCorners)
                pts_prev.resize(maxCorners);
        }

        int k = waitKey(10);

        if (k == 27 || k == 'q')
            break;
    }

    return 0;
}
