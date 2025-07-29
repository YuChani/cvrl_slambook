#include <rerun.hpp>
#include <rerun/demo_utils.hpp>

using namespace std;
using namespace rerun::demo;
namespace rr = rerun;

int main() 
{
    const auto rec = rerun::RecordingStream("rerun_example_cpp");

    rec.spawn().exit_on_failure();

    vector<rerun::Position3D> points = grid3d<rr::Position3D, float>(-10.f, 10.f, 10);
    vector<rerun::Color> colors = grid3d<rr::Color, uint8_t>(0, 255, 10);

    rec.log("my_points", rerun::Points3D(points).with_colors(colors).with_radii({0.5f}));
}