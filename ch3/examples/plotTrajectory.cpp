#include <iostream>
#include <Eigen/Core>  
#include <unistd.h>
#include <pangolin/pangolin.h>

using namespace std;
using namespace Eigen;  

// path trajectory file path
const string trajectory_file = "/home/chani/personal/slambook2/ch3/examples/trajectory.txt";

void DrawTrajectory(vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>>);

int main(int argc, char **argv)
{
    // trajectory include vector. each pose is represented as an Isometry3d
    vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>> poses;
    ifstream fin(trajectory_file);

    // if not find file print "file error"
    if(!fin)
    {
        cout << "file error" << trajectory_file << endl;
        return 1;
    }

    // while end of file, read each line. each line contain pose(time, positoin, qaternion)
    while(!fin.eof())
    {
        double time, tx, ty, tz, qx, qy, qz, qw;
        fin >> time >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
        Isometry3d w_T_r(Quaterniond(qw, qx, qy, qz));
        w_T_r.pretranslate(Vector3d(tx, ty, tz));
        poses.push_back(w_T_r);
    }

    cout << "read total " << poses.size() << " poses" << endl;

    // draw trajectory
    DrawTrajectory(poses);

    return 0;
}

void DrawTrajectory(vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>> poses)
{
    pangolin::CreateWindowAndBind("Tragectory Viewer", 1024, 768);  // pangolin window 생성
    // pangolin Rendering option setting(depth, test, blending)
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Define Camera Render Object (for 3D rendering)
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 420, 420, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    // mouse handler for camera control
    pangolin::View &d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));

    // while pangolin windown closed repeat
    while(pangolin::ShouldQuit() == false)
    {
        // set backgroun color / line width
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glLineWidth(2);

        // draw pangolin trajectory
        for (size_t i = 0; i < poses.size(); i++)
        {   
            // calculate pose (position and orientation)
            Vector3d Ow = poses[i].translation();
            Vector3d Xw = poses[i] * (0.1 * Vector3d(1, 0, 0));
            Vector3d Yw = poses[i] * (0.1 * Vector3d(0, 1, 0));
            Vector3d Zw = poses[i] * (0.1 * Vector3d(0, 0, 1));

            // draw coordinate frame
            glBegin(GL_LINES);
            glColor3f(1.0, 0.0, 0.0); // Red for X axis
            glVertex3d(Ow[0], Ow[1], Ow[2]);
            glVertex3d(Xw[0], Xw[1], Xw[2]);

            glColor3f(0.0, 1.0, 0.0); // Blue for Y axis
            glVertex3d(Ow[0], Ow[1], Ow[2]);
            glVertex3d(Yw[0], Yw[1], Yw[2]);

            glColor3f(0.0, 0.0, 1.0); // Green for Z axis
            glVertex3d(Ow[0], Ow[1], Ow[2]);
            glVertex3d(Zw[0], Zw[1], Zw[2]);

            glEnd();
        }

        // draw trajectory line
        for(size_t i = 0; i < poses.size() - 1; i++)
        {
            glColor3f(0.0, 0.0, 0.0);
            glBegin(GL_LINES);

            auto p1 = poses[i], p2 = poses[i + 1];

            glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
            glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
            glEnd();
        }

        pangolin::FinishFrame();

        usleep(5000); // sleep 5ms
    }
}