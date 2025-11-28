#include "include.h"
#include "DataSet.h"
#include "IQHV.h"
using namespace qhv;
int main()
{
    // TPoint* p1 =  new TPoint(4);
    // ifstream stream("testPoint2");
    // if(stream.fail())
    //     std::cout << "No such file";
    // p1->Load(stream);
    // std::cout<<*p1;
    // stream.close();

    // string a = "abc";
    // vector<string> g = split(a,"b");
    // for(auto tok : g)
    // {
    //     cout << tok;
    // }

    // OptimizationProblemSettings *ops = new OptimizationProblemSettings("moja_nazwa_eksperymentu_d7n200_3"); 

    // TOptimizationProblem top = TOptimizationProblem("../../uniform_sphere/uniform_sphere_d4n100_1", true);

    // cout << top.points.at(0)->ObjectiveValues[0] << endl;
    // cout << (*top.points[0])[0];
    auto all_problems = DataSet::LoadBulk("C:/Users/kubad/hypervolume/problems/uniform_sphere/");
   /* auto grouped_problems = DataSet::BulkGroup(all_problems, ProblemGrouping::Dimensionality);*/
    /*DataSet::printGroupsDetails(grouped_problems);*/
    IQHV solver;

    // cout << solver.Solve(all_problems[110]).hypervolume << endl;
    for (auto problem = begin(all_problems); problem != end(all_problems); problem++)
    {
        auto result = solver.Solve(&(*problem));

        cout << problem->filename << " " << ((HypervolumeResult*)result)->Volume << endl;
    }
    //     cout<< all_problems[110].filename << endl;
        // cout<<problem->filename << endl;
        // s.Solve(*problem);

}