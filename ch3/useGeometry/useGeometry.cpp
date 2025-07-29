#include <iostream>
#include <ctime>
#include <Eigen/Core>   // Eigen Core
#include <Eigen/Geometry> 

using namespace Eigen;
using namespace std;

int main(int argc, char **argv)
{   
    // Rotation Marix 3x3 생성 Eigen::Matrix3d
    Matrix3d rotation_matrix = Matrix3d::Identity();    // 3x3 단위행렬 생성
    // Rotation vector 3x1 생성 Eigen::AngleAxisd
    // z축을 기준으로 45도 회전 Vector3d(x, y, z)
    AngleAxisd rotation_vector(M_PI / 4, Vector3d(0, 0, 1));    // Rotatce 45 degrees along z axis
    
    cout.precision(3);  // 자릿수 고정
    cout << "rotation matrix = \n" << rotation_vector.matrix() << endl; 

    rotation_matrix = rotation_vector.toRotationMatrix();   // Axis-angle convert to rotation matrix
    
    Vector3d v(1, 0, 0);

    Vector3d v_rotated = rotation_vector * v;   
    cout << "(1, 0, 0) after rotation by axis angle = " << v_rotated.transpose() << endl;    // rotation vector (axis angle)

    v_rotated = rotation_matrix * v;
    cout << "(1, 0, 0) after rotation by Matrix     = " << v_rotated.transpose() << endl;    // rotation matrix

    Vector3d euler_angles = rotation_matrix.eulerAngles(2, 1, 0);   // ZYX order (Roll, Pitch, Yaw)
    cout << "Yaw Pitch Roll (ZYX order)             = " << euler_angles.transpose() << endl;

    Isometry3d T = Isometry3d::Identity();  // 3D 변환 행렬
    T.rotate(rotation_vector);
    T.pretranslate(Vector3d(1, 3, 4));
    cout << "Transform matrix = \n" << T.matrix() << endl;;

    Vector3d v_transformed = T * v;
    cout << "v transformed                   = " << v_transformed.transpose() << endl;

    Quaternion q = Quaterniond(rotation_vector);
    cout << "Quaternion from rotation vector = " << q.coeffs().transpose() << endl;

    q = Quaterniond(rotation_matrix);
    cout << "Quaternion from rotation matrix = " << q.coeffs().transpose() << endl;

    v_rotated = q * v;
    cout << "(1, 0 , 0) after rotation       = " << v_rotated.transpose() << endl;

    cout << "should be equal to              = " <<  (q * Quaterniond(0, 1, 0, 0) * q.inverse()).coeffs().transpose() << endl;


    return 0;
}
