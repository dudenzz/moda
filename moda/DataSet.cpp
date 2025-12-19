#include "DataSet.h"
#include <stdexcept>
namespace moda {
    DataSet::DataSet(const DataSet& dataSet)
    {
        
        nadir = new Point(*const_cast<DataSet*>(&dataSet)->getNadir());
        ideal = new Point(*const_cast<DataSet*>(&dataSet)->getIdeal());
        presetParameters = false;
        NormalizingFunction = dataSet.NormalizingFunction;
        typeOfOptimization = dataSet.typeOfOptimization;

        for (auto point : dataSet.points)
            points.push_back(new Point(*point));

        parameters.filename = dataSet.parameters.filename;
        parameters.name = dataSet.parameters.name;
        parameters.nPoints = dataSet.parameters.nPoints;
        parameters.NumberOfObjectives = dataSet.parameters.NumberOfObjectives;
        parameters.optType = dataSet.parameters.optType;
        parameters.sampleNumber = dataSet.parameters.sampleNumber;

    }
    std::vector<Point*> DataSet::defaultScalingFuncation(std::vector <Point*> points)
    {
        if (points.size() == 1) {
            for (auto point : points)
                for (int i = 0; i < point->NumberOfObjectives; i++)
                {
                    if (this->typeOfOptimization == OptimizationType::maximization)
                        (*point)[i] = 1;
                    if (this->typeOfOptimization == OptimizationType::minimization)
                        (*point)[i] = 0;
                }
            return points;
        }
        Point* maximumPoint = new Point(*points.at(0));
        Point* minimumPoint = new Point(*points.at(0));
        for (auto point : points)
        {
            for (int i = 0; i < point->NumberOfObjectives; i++)
            {
                //jeżeli problem jest minimalizowany wtedy zmienia się "znak" operacji logicznej na przeciwny
                if ((*point)[i] > (*maximumPoint)[i])
                    (*maximumPoint)[i] = (*point)[i];
                if ((*point)[i] < (*minimumPoint)[i])
                    (*minimumPoint)[i] = (*point)[i];
            }
        };
        for (auto point : points)
            for (int i = 0; i < point->NumberOfObjectives; i++)
            {
                (*point)[i] = ((DType)(*point)[i] - (*minimumPoint)[i]) / ((*maximumPoint)[i] - (*minimumPoint)[i]);
            }
        delete maximumPoint;
        delete minimumPoint;
        return points;
    }
    /** getter */
    Point* DataSet::operator[](int n) const
    {
        return points.at(n);
    }
    /** getter */
    Point* DataSet::get(int n) const
    {
        return points.at(n);
    }
    /** setter */
    Point& DataSet::operator[](int n)
    {
        return *points.at(n);
    }
    void DataSet::normalize()
    {
        points = defaultScalingFuncation(points);
        calculateIdeal();
        calculateNadir();

    }



    Point* DataSet::defaultWorsePoint()
    {        
        Point* better = new Point(*getIdeal());
        Point* worse = new Point(*getNadir());
        for (int i = 0; i < parameters.NumberOfObjectives; i++)
        {
            (*worse)[i] = (*worse)[i] - ((*better)[i] - (*worse)[i]) * 0.1;
        }
        return worse;

    }
    Point* DataSet::defaultBetterPoint()
    {
        Point* better = new Point(*getIdeal());
        Point* worse = new Point(*getNadir());
        for (int i = 0; i < parameters.NumberOfObjectives; i++)
        {
            (*better)[i] = (*better)[i] + ((*worse)[i] - (*better)[i]) * 0.1;
        }
        return better;

    }
    Point* DataSet::getNadir()
    {
        calculateNadir();
        return nadir;
    }

    Point* DataSet::getIdeal()
    {
        calculateIdeal();
        return ideal;
    }
    void DataSet::RemoveDominated()
    {

        //for (auto i = points.begin(); i <= points.end(); i++)
        //{
        //    if (indexSet[i] != NULL)
        //    {
        //        int i2;
        //        for (i2 = i + 1; i2 <= end; i2++)
        //        {
        //            if (indexSet[i2] != NULL && indexSet[i] != NULL)
        //            {
        //                bool better = false;
        //                bool better = false;
        //                short j;
        //                for (j = 0; j < getParameters().NumberOfObjectives && !(better && better); j++)
        //                {
        //                    DType o1 = min(indexSet[i]->get(j), ideal[j]);

        //                    DType o2 = min(indexSet[i2]->get(j), nadir[j]);

        //                    better = better || (o1 > o2);
        //                    better = better || (o1 < o2);
        //                }
        //                if (!better)
        //                {
        //                    // i2 covered
        //                    indexSet[i2] = indexSet[end--];
        //                }
        //                if (better && !better)
        //                {
        //                    // i dominated
        //                    indexSet[i] = indexSet[end--];
        //                }
        //            }
        //        }
        //    }
        //}
    }
    void DataSet::setIdeal(Point* newIdeal)
    {
        delete ideal;
        ideal = newIdeal;
    }

    void DataSet::setNadir(Point* newNadir)
    {
        delete nadir;
        nadir = newNadir;
    }
    //calculateIdeal
    void DataSet::calculateIdeal() {
        delete ideal;
        ideal = new Point(*points.at(0));
        for (auto point : points)
        {
            for (int i = 0; i < parameters.NumberOfObjectives; i++)
            {
                if ((*point)[i] > (*ideal)[i])
                    (*ideal)[i] = (*point)[i];
            }
        }
       
    }

    //calculateNadir
    void DataSet::calculateNadir() {
        delete nadir;
        nadir = new Point(*points.at(0));
        for (auto point : points)
        {
            for (int i = 0; i < parameters.NumberOfObjectives; i++)
            {
                if ((*point)[i] < (*nadir)[i])
                    (*nadir)[i] = (*point)[i];
            }
        }
    }

    DataSet::DataSet(int nObjectives)
    {
        nadir = NULL;
        ideal = NULL;
        presetParameters = false;
        //NormalizingFunction = (DataSet::defaultScalingFuncation);
        typeOfOptimization = OptimizationType::maximization; 
        parameters.nPoints = 0;
        parameters.NumberOfObjectives = nObjectives;
    }


    // TOptimizationProblem::TOptimizationProblem(const std::string filename)
    // {

    // }

    DataSet::DataSet(const std::string filename, bool normalizedName)
    {
        nadir = NULL;
        ideal = NULL;
        this->filename = filename;
        if (normalizedName)
        {
            parameters = DataSetParameters(filename);
            presetParameters = true;
        }
        std::ifstream stream = std::ifstream(filename);
        Load(stream);


    }

    DataSet::DataSet(const std::string filename, DataSetParameters settings)
    {
        nadir = NULL;
        ideal = NULL;
        this->filename = filename;
        settings = settings;
        presetParameters = true;
        std::ifstream stream = std::ifstream(filename);
        Load(stream);
    }    

    DataSet::DataSet(const std::string filename, std::string name, int dimensions, int sampleNumber, int nPoints)
    {
        nadir = NULL;
        ideal = NULL;
        this->filename = filename;
        parameters = DataSetParameters(name, dimensions, nPoints, sampleNumber);
        presetParameters = true;
        std::ifstream stream = std::ifstream(filename);
        Load(stream);
    }

    void DataSet::Save(const std::string filename)
    {
        std::ofstream stream(filename);
        for (auto point : points)
        {
            for (int i = 0; i < point->NumberOfObjectives; i++)
            {
                stream << point->ObjectiveValues[i] << " ";
            }
            stream << std::endl;
        }
        stream.close();
    }

    std::istream& DataSet::Load(std::istream& stream)
    {
        std::string line = "";

        int no_objectives = -1;
        int no_points = 0;
        std::vector<DType> values;
        while (std::getline(stream, line))
        {
            line = trim(line);
            if (line == "")
                continue;
            if (line == "#")
                break;
            bool contains_space = line.find(" ") != std::string::npos;
            bool contains_comma = line.find(",") != std::string::npos;
            bool contains_tabulator = line.find("\t") != std::string::npos;
            //dodać średnik
            if (!(contains_space || contains_comma || contains_tabulator))
                throw new std::runtime_error("No separator (delimiter) found in file. Avaliable separators: ' '(space), ','(comma), '\t'(tabulator). The library does not process single objective problems.");//something is wrong, no separator found
            if ((contains_comma && contains_space) || (contains_comma && contains_tabulator) || (contains_space && contains_tabulator))
                throw new std::runtime_error("Multiple separators (delimiters) found in file. Avaliable separators: ' '(space), ','(comma), '\t'(tabulator). Normalize the separator usage in file, make sure you are using the same separator every time.");//something is wrong, multiple separators found
            std::string separator = contains_comma ? "," : contains_space ? " " : "\t";
            std::string token;
            int no_objectives_in_this_line = 0;
            values.clear();
            size_t pos = 0;
            while ((pos = line.find(separator)) != std::string::npos)
            {
                token = line.substr(0, pos);
                line.erase(0, pos + 1);
                std::string::iterator end_pos = std::remove(token.begin(), token.end(), ' ');
                token.erase(end_pos, token.end());
                end_pos = std::remove(token.begin(), token.end(), '\t');
                token.erase(end_pos, token.end());
                #if DTypeN == 2 // if the datatype is DType (else it's float)
                values.push_back(std::stod(token));
                #else
                values.push_back(std::stof(token));
                #endif
                no_objectives_in_this_line += 1;
            }
            token = line;
            std::string::iterator end_pos = std::remove(token.begin(), token.end(), ' ');
            token.erase(end_pos, token.end());
            end_pos = std::remove(token.begin(), token.end(), '\t');
            token.erase(end_pos, token.end());
            #if DTypeN == 2 // if the datatype is DType (else it's float)
            values.push_back(std::stod(token));
            #else
            values.push_back(std::stof(token));
            #endif
            no_objectives_in_this_line += 1;



            if (no_objectives == -1) no_objectives = no_objectives_in_this_line;
            if (no_objectives_in_this_line != no_objectives) throw new std::runtime_error("Wrong number of objectives in one of the points in the file. Check the file.");
            Point* p = new Point(no_objectives);
            for (int i = 0; i < no_objectives; i++)
                p->ObjectiveValues[i] = values.at(i);
            
            this->add(p);
            no_points++;
    
        }
        this->parameters.NumberOfObjectives = no_objectives;
        this->parameters.nPoints = no_points;

        return stream;
    }

    DataSet::~DataSet()
    {
        for (auto point : points)
        {
            if (point != NULL && point->NumberOfObjectives > 0 && point->NumberOfObjectives < 20)
                delete point;
        }
        this->points.clear();
        this->points.shrink_to_fit();
        delete ideal;
        delete nadir;

    }

    DataSet& DataSet::operator=(DataSet& problem)
    {
        this->points.clear();
        for (auto point : problem.points)
            this->points.push_back(point);
        return *this;
    }


    DataSetParameters* DataSet::getParameters()
    {
        return new DataSetParameters(parameters);
    }


    void DataSet::setParameters(DataSetParameters settings)
    {
        presetParameters = true;
        this->parameters = settings;
    }



    void DataSet::setDimensionality(int dim)
    {
        parameters.NumberOfObjectives = dim;
    }


    void DataSet::setName(std::string name)
    {
        parameters.name = name;
    }


    void DataSet::setNumberOfPoints(int npts)
    {
        parameters.nPoints = npts;
    }


    void DataSet::setSampleNumber(int sampleN)
    {
        parameters.sampleNumber = sampleN;
    }


    DataSet* DataSet::LoadFromFilename(const std::string filename)
    {
        
        DataSet* problem = new DataSet();

        //problem->setParameters(DataSetParameters(filename));
        problem->typeOfOptimization = DataSet::OptimizationType::maximization;
        problem->filename = filename;
        problem->nadir = new Point(problem->getParameters()->NumberOfObjectives);
        problem->ideal = new Point(problem->getParameters()->NumberOfObjectives);
        std::ifstream stream = std::ifstream(filename);
        problem->Load(stream);
        stream.close();
        return problem;
    }
    std::string DataSet::to_string()
    {
        std::stringstream ss;
        for (int i = 0; i < getParameters()->nPoints; i++)
        {
            for (int j = 0; j < getParameters()->NumberOfObjectives; j++)
            {
                ss << points.at(i)->get(j) << " ";
            }
            ss << std::endl;
        }
        return ss.str();
    }
    //ta metoda powinna mieć wersje z rozszerzeniem pliku (maską)
    std::vector<DataSet*> DataSet::LoadBulk(const std::string directory)
    {

        std::chrono::time_point start = std::chrono::high_resolution_clock::now();

        float fileCount = (float)countFilesInDirectory(directory);
        int currentFile = 0;

        std::vector<DataSet*> problems;
        for (const auto& entry : std::filesystem::directory_iterator(directory))
        {
            std::string pth = entry.path().string();
            try {
                problems.push_back(DataSet::LoadFromFilename(pth));
            }
            catch (std::exception ex) {
                std::cout << "Couldn't load file (file skipped): " << pth << std::endl;
                std::cout << "Exception message: " << ex.what() << std::endl;
            }
            currentFile += 1;
            make_progress(currentFile / fileCount, currentFile, (int)fileCount);
        }
        std::cout << std::endl;
        std::chrono::time_point end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Files loaded in " << duration.count() << "ms." << std::endl;
        return problems;
    }


    bool DataSet::add(Point* newPoint)
    {
        parameters.nPoints += 1;
        parameters.NumberOfObjectives = newPoint->NumberOfObjectives;
        points.push_back(newPoint);
        calculateIdeal();
        calculateNadir();
        return true;
    }

    bool DataSet::remove(Point* toRemove)
    {
        std::vector<Point*>::const_iterator where;
        bool found = false;
        for (std::vector<Point*>::iterator iterator; iterator != points.end(); iterator++)
        {
            bool theSame = true;
            for (int i = 0; i < toRemove->NumberOfObjectives; i++)
            {
                if (toRemove->ObjectiveValues[i] != (*iterator)->ObjectiveValues[i]) theSame = false;
            }
            if (theSame)
            {
                where = iterator;
                found = true;
                break;
            }
        }
        if (found) points.erase(where);
        calculateNadir();
        calculateIdeal();
        return found;
    }
    Point* DataSet::remove(int where)
    {
        Point* found = new Point(this->parameters.NumberOfObjectives);
        std::vector<Point*>::iterator iterator = points.begin();
        for (int temp = 0; temp <= where; temp++)
        {
            iterator++;
            found = new Point(*points.at(temp));
        }
        iterator--;
        points.erase(iterator);
        calculateNadir();
        calculateIdeal();
        parameters.nPoints--;
        return found;
    }
    void DataSet::clear()
    {
        points.clear();
        parameters.nPoints = 0;
    }
    int getIndex(std::vector<int> v, int K)
    {
        auto it = find(v.begin(), v.end(), K);
        int index = -1;
        if (it != v.end())
        {
            index = it - v.begin();
        }
        return index;
    }


    std::vector<std::vector<DataSet>> DataSet::BulkGroup(std::vector<DataSet> problems, ProblemGrouping grouping)
    {
        std::vector<std::vector<DataSet>> groupedProblems;
        std::vector<int> dimensionalities;
        switch (grouping)
        {
        case ProblemGrouping::Dimensionality:

            for (auto problem : problems)
            {
                if (std::find(dimensionalities.begin(), dimensionalities.end(), problem.getParameters()->NumberOfObjectives) == dimensionalities.end())
                {
                    dimensionalities.push_back(problem.getParameters()->NumberOfObjectives);
                }
            }
            for (auto _ : dimensionalities)
            {
                groupedProblems.push_back(std::vector<DataSet>());
            }
            for (auto problem : problems)
            {
                int index = getIndex(dimensionalities, problem.getParameters()->NumberOfObjectives);
                groupedProblems[index].push_back(problem);
            }
            break;
        case ProblemGrouping::Name:
            break;
        case ProblemGrouping::NameDimensionality:
            break;
        default:
            break;
        }
        return groupedProblems;
    }

    void DataSet::reverseObjectives()
    {

        for (int i = 0; i < parameters.nPoints; i++) {
            if (points[i] == NULL) {
                continue;
            }
            *points[i] = 1.0f - *points[i];
            //short j;
            //for (j = 0; j < parameters.NumberOfObjectives; j++) {
            //    
            //    //TODO : ma wyglądać tak:
            //    points[i][j] = 1 - points[i][j];
            //    
            //}
        }
    }


    void DataSet::printGroupsDetails(std::vector<std::vector<DataSet>> groupedProblems)
    {
        int c_number = 1;
        for (auto group : groupedProblems)
        {
            auto names = std::vector<std::string>();
            auto dims = std::vector<int>();
            auto nPoints = std::vector<int>();
            int count = 0;

            for (auto problem : group)
            {
                if (std::find(dims.begin(), dims.end(), problem.getParameters()->NumberOfObjectives) == dims.end())
                    dims.push_back(problem.getParameters()->NumberOfObjectives);
                if (std::find(names.begin(), names.end(), problem.getParameters()->name) == names.end())
                    names.push_back(problem.getParameters()->name);
                if (std::find(nPoints.begin(), nPoints.end(), problem.getParameters()->nPoints) == nPoints.end())
                    nPoints.push_back(problem.getParameters()->nPoints);
                count += 1;
            }
            std::cout << "Group " << c_number << std::endl;
            std::cout << "\tNames : ";
            for (auto name : names)
                std::cout << name << " ; ";
            std::cout << std::endl
                << "\tNumber of Objectives : ";
            for (auto dim : dims)
                std::cout << dim << " ; ";
            std::cout << std::endl
                << "\tNumbers of Points : ";
            for (auto nPoint : nPoints)
                std::cout << nPoint << " ; ";
            std::cout << std::endl
                << "\tNumbers of Datasets : " << count << std::endl
                << std::endl;
            c_number++;
        }
    }
    
    NDTree<Point> DataSet::toNDTree() {
        NDTree<Point> NDTree;
        for (auto p : points)
        {
            NDTree.update(*p);
        }
        return NDTree;
    }

}