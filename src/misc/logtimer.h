#ifndef __LOGTIMER__
#define __LOGTIMER__

#include <time.h>
#include <vector>
/**
 * Timer class records statistics based on the speed and counts. 
 *
 * Main usage is recording how long different steps of ray tracing
 * takes to determine areas that we can speed up. 
 */ 
namespace CGL {
class LogTimer{
    public:

        LogTimer(){};
        ~LogTimer(){};

       
        void startTime(int index);
        void recordTime(int index);
        void recordCount(int index);

        double getTime(int index);
        int    getCount(int index);
       

        //void upsample  ( HalfedgeMesh& mesh );
        //void downsample( HalfedgeMesh& mesh );
        //void resample  ( HalfedgeMesh& mesh );
    private:
        std::vector<double> section_time;
        std::vector<double> start_time;
        std::vector<int> count;

        int addTimer();
        int addCounter();


        time_t timer;
};

} //CGL
#endif  //LOGTIMER