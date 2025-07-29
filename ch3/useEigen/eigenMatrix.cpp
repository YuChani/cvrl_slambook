#include <iostream>
#include <ctime>
#include <Eigen/Core>   // Eigen Core
#include <Eigen/Dense>  // Algebra operation of dense Matrix

#define MATRIX_SIZE 50

using namespace Eigen;
using namespace std;

int main(int argc, char **argv)
{
    // declare Matrix 2x3
    Matrix<float, 2, 3> matrix_23; 

    Vector3d v_3d;
    Matrix<float, 3, 1> vd_3d;

    // Matrix3d is essentially Eigen::Matrix
    Matrix3d matrix_33 = Matrix3d::Zero();  // initialized to zero
    // dynamic size
    Matrix <double, Dynamic, Dynamic> matirx_dynamic;
    // simpler
    MatrixXd matrix_x;

    matrix_23 << 1, 2, 3, 4 ,5, 6;

    cout << "matrix 2x3 1 ~ 6 : \n" << matrix_23 << endl;
    
    cout << "matrix 2x3 : " << endl;

    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            cout << matrix_23(i, j) << "\t";
        }
        cout << endl;
    }

    v_3d << 3, 2, 1;
    vd_3d << 4, 5, 6;

    Matrix<double, 2, 1> result = matrix_23.cast<double>() * v_3d;
    cout << "[1, 2, 3, 4, 5, 6]*[4, 5, 6] : " << result.transpose() << endl;    

    Matrix<float, 2, 1> result2 = matrix_23 * vd_3d;
    cout << "[1, 2, 3, 4, 5, 6]*[4, 5, 6] : " << result2.transpose() << endl;    

    matrix_33 = Matrix3d::Random(); // Random Number Matrix
    cout << "random : \n" << matrix_33 << "\n" << endl;
    cout << "transpose : \n" << matrix_33.transpose() << "\n" << endl;
    cout << "sum : \n" << matrix_33.sum() << "\n" << endl;
    cout << "trace : \n" << matrix_33.trace() << "\n" << endl;
    cout << "times 10 : \n" << 10 * matrix_33 << "\n" << endl;
    cout << "inverse : \n" << matrix_33.inverse() << "\n" << endl;
    cout << "det : \n" << matrix_33.determinant() << "\n" << endl;

    // Eigenvalues
    SelfAdjointEigenSolver<Matrix3d> eigen_solver(matrix_33.transpose() * matrix_33);
    cout << "Eigne values = \n" << eigen_solver.eigenvalues() <<  "\n" <<endl;
    cout << "Eigen vectors = \n" << eigen_solver.eigenvectors() <<  "\n" <<endl;

    Matrix<double, MATRIX_SIZE, MATRIX_SIZE> matrix_NN = MatrixXd::Random(MATRIX_SIZE, MATRIX_SIZE);
    matrix_NN = matrix_NN * matrix_NN.transpose();  // Guaratee semi-positive definite
    Matrix<double, MATRIX_SIZE, 1> v_Nd = Matrix<double, MATRIX_SIZE, 1>::Random();

    clock_t time_stt = clock(); 

    Matrix<double, MATRIX_SIZE, 1> x = matrix_NN.inverse() * v_Nd;
    cout << "time of normal inverse is " << 1000 * (clock() - time_stt) / (double) CLOCKS_PER_SEC << "ms" << endl;
    cout << "x = " << x.transpose() << "\n" << endl;


    time_stt = clock();
    x = matrix_NN.colPivHouseholderQr().solve(v_Nd);
    cout << "time of Qr decomposition is " << 1000 * (clock() - time_stt) / (double) CLOCKS_PER_SEC << "ms" << endl;
    cout << "x = " << x.transpose() << "\n" << endl;


    time_stt = clock();
    x = matrix_NN.ldlt().solve(v_Nd);
    cout << "time of ldlt decomposition is " << 1000 * (clock() - time_stt) / (double) CLOCKS_PER_SEC << "ms" << endl;
    cout << "x = " << x.transpose() << "\n" << endl;

    return 0;
}