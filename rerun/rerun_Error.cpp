#include <iostream>
#include <fstream>
#include <vector>
#include <thread>       // sleep 사용
#include <chrono>       // chrono::milliseconds

#include <rerun.hpp>
#include <rerun/datatypes/quaternion.hpp>
#include <rerun/archetypes.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace std;
using namespace rerun::components;
using namespace rerun::datatypes;
using namespace rerun::archetypes;

struct Pose
{
    double timestamp;
    Eigen::Vector3d t;
    Eigen::Quaterniond q;
};

// Trajectory 로드
vector<Pose> LoadTrajectory(const string& path, bool w_first = false) 
{
    vector<Pose> traj;
    ifstream fin(path);

    if (!fin.is_open()) 
    {
        cerr << "Cannot open " << path << endl;
        return traj;
    }

    while (!fin.eof()) 
    {
        Pose p;
        double tx, ty, tz, q1, q2, q3, q4;
        fin >> p.timestamp >> tx >> ty >> tz >> q1 >> q2 >> q3 >> q4;

        if(fin.fail()) 
        {
            break;
        }
        
        p.t = Eigen::Vector3d(tx, ty, tz);

        if (w_first)
        {
            p.q = Eigen::Quaterniond(q1, q2, q3, q4); // wxyz
        }
        
        else
        {
            p.q = Eigen::Quaterniond(q4, q1, q2, q3); // xyzw -> wxyz
        }

        traj.push_back(p);
    }

    return traj;
}


int main() 
{
    auto gt_traj  = LoadTrajectory("/home/chani/personal/slambook2/ch4/example/groundtruth.txt", true);  // wxyz
    auto est_traj = LoadTrajectory("/home/chani/personal/slambook2/ch4/example/estimated.txt", false);  // xyzw

    cout << "Loaded " << gt_traj.size() << " GT poses, " << est_traj.size() << " estimated poses\n";

    rerun::RecordingStream rec("Trajectory Viewer");
    rec.spawn().exit_on_failure();

    // Groundtruth 
    {
        std::vector<Vector3D> gt_points;
        for(auto& p : gt_traj) 
            gt_points.emplace_back(p.t.x(), p.t.y(), p.t.z());

        LineStrip3D gt_strip(gt_points);
        std::vector<LineStrip3D> gt_collection{gt_strip};

        rec.log("trajectory/groundtruth",
            LineStrips3D(gt_collection)
                .with_colors({Color(255,0,0)})  // Red 
        );
    }

    // Estimated trajectory
    {
        std::vector<Vector3D> est_points;

        for(size_t i = 0; i < est_traj.size(); ++i) 
        {
            auto& p = est_traj[i];
            est_points.emplace_back(p.t.x(), p.t.y(), p.t.z());


            rec.log("trajectory/estimated_axis",
                Transform3D()
                    .with_translation(Translation3D(p.t.x(), p.t.y(), p.t.z()))
                    .with_rotation(Quaternion::from_xyzw(
                        static_cast<float>(p.q.x()), 
                        static_cast<float>(p.q.y()), 
                        static_cast<float>(p.q.z()), 
                        static_cast<float>(p.q.w())
                    ))
                    .with_axis_length(0.5f)
            );


            LineStrip3D est_strip(est_points);
            vector<LineStrip3D> est_collection{est_strip};
            rec.log("trajectory/estimated",
                LineStrips3D(est_collection)
                    .with_colors({Color(0,0,255)})  // Blue
            );

            this_thread::sleep_for(std::chrono::milliseconds(30)); 
        }
    }

    cout << "Rerun Viewer \n";


    this_thread::sleep_for(std::chrono::seconds(2));

    return 0;
}
