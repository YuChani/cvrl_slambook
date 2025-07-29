#include <iostream>
#include <vector>
#include <algorithm>
#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace std;
using namespace Eigen;

int main(int argc, char **argv)
{
    // Quaterniond : w + xi + yj + zk
    Quaterniond q1(0.35, 0.2, 0.3, 0.1), q2(-0.5, 0.4, -0.1, 0.2);
    // Quaternion normalization
    q1.normalize(); 
    q2.normalize();
    // R1, R2의 translation - world 좌표계 기준
    Vector3d t1(0.3, 0.1, 0.1), t2(-0.1, 0.5, 0.3);
    // R1 frame에서 관측한 p1의 좌표
    Vector3d p1(0.5, 0, 0.2);

    // Isometry3d : SE(3) 표현 translation은 없는 상태
    Isometry3d R1_T1_W(q1), R2_T2_W(q2);
    // translation 적용
    R1_T1_W.pretranslate(t1);
    R2_T2_W.pretranslate(t2);

    // R2_P2 = R2_T2_W * W_T_R1 * R1_P1;
    Vector3d p2 = R2_T2_W * R1_T1_W.inverse() * p1;
    // tanspose는 세로로된 행렬을 가로로 출력
    cout << endl << p2.transpose() << endl;

    return 0;
}


// target_T_source : source 좌표계에 있는 점을 target 좌표계 기준으로 변환