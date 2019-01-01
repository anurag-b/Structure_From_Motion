/**
 *@file main.cpp
 *@author Anurag Bansal
 *@brief Structure from Motion in C++
 */

//#include "elas.h"
#include "SFM.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv){
	//Read all the images in a vector
	const char* calib_file_name = "../calibration/kitti.yml";
	vector<String> images_path;
	glob("../images/*.JPG",images_path,false);
	SFM exe(calib_file_name);
	exe.execute(images_path);
}