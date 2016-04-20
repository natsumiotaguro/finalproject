
#include "logtimer.h"

namespace CGL {



void LogTimer::startTime(int index){
	if(index >= start_time.size()){
		addTimer();
	}
	else{
		start_time[index] = time(&timer); 
	}
	
}

void LogTimer::recordTime(int index){
	double difference = difftime(time(&timer), start_time[index]);
	section_time[index] += difference;
}

void LogTimer::recordCount(int index){
	if(index >= count.size()){
		addCounter();
	}
	count[index]++;
}

double LogTimer::getTime(int index){
	return section_time[index];
}

int    LogTimer::getCount(int index){
	return count[index];
}

int LogTimer::addTimer(){
	section_time.push_back(0);
	start_time.push_back(time(&timer));
}

int LogTimer::addCounter(){
	count.push_back(0);
}


}