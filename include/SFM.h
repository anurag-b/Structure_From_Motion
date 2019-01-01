/**
 *@file SFM.h
 *@author Anurag Bansal
 *@brief Structure from Motion in C++
 */

#include <iostream>
#include <cstdint>
#include <string>

#include <sys/types.h>
//#include "STLReader/CSTLReader.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/core/persistence.hpp"
#include "opencv2/video/tracking.hpp"
#include <opencv2/features2d/features2d.hpp>

#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/file_io.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>

#define EPSILON 0.00001

class SFM{
private:
	cv::Mat img, img_gray, img_curr, img_prev;
	cv::Mat F,E;
	std::vector<cv::Point2f> sel_points, tracked_points;
	cv::FileStorage calib_file;
	cv::Mat K, D;
	//std::vector<cv::CloudPoint> pointcloud
	std::vector<cv::Point3f> pointcloudXYZ;
	//Point-cloud pointer initialised for a XYZRGB pointcloud storage
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr {new pcl::PointCloud<pcl::PointXYZRGB>};

	void populatePCLPointCloud(const std::vector<cv::Point3f> pointcloud,bool write_to_file);
	void detectFeatures(cv::Mat img);
	void trackFeatures(cv::Mat img_curr, cv::Mat img_prev);
	void getFundamentalMatrix();
	bool getProjectionMatrix();
	double triangulate(cv::Matx34d P1, cv::Matx34d P2);
	cv::Mat_<double> IterativeLinearLSTriangulation(cv::Point3d u1, cv::Matx34d P1, cv::Point3d u2, cv::Matx34d P2);
	cv::Mat_<double> LinearLSTriangulation(cv::Point3d u1, cv::Matx34d P1, cv::Point3d u2, cv::Matx34d P2);
	bool testTriangulation(const cv::Matx34d& P); //chirelity condition/coplanarity constraint check
public:
	SFM(const char* calib_file_name);
	void execute(std::vector<cv::String> images_path);
};