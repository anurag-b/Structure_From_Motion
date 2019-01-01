/**
 *@file SFM.cpp
 *@author Anurag Bansal
 *@brief Structure from Motion in C++
 */

#include "SFM.h"

using namespace std;
using namespace cv;

void SFM::populatePCLPointCloud(vector<Point3f> pointcloud,bool write_to_file){
	//Populate point cloud
	cout << "Creating point cloud...";
	//double t = cv::getTickCount();
	double px,py,pz;
	uchar pr,pg,pb;
	for (unsigned int i=0; i<pointcloud.size(); i++) {
		// get the RGB color value for the point
		cv::Vec3b rgbv(255,255,255);
		Point3_<uchar>* p = img.ptr<Point3_<uchar> >(sel_points[i].y,sel_points[i].x);
	
		px = pointcloud[i].x;
		py = pointcloud[i].y;
		pz = pointcloud[i].z;

		pb = p->x;
		pg = p->y;
		pr = p->z;

		pcl::PointXYZRGB point;
		
		// 3D coordinates
		point.x = px;
		point.y = py;
		point.z = pz;
		

		// RGB color, needs to be represented as an integer
		std::uint32_t rgb = (static_cast<std::uint32_t>(pr)<<16 | static_cast<std::uint32_t>(pg)<<8
							| static_cast<std::uint32_t>(pb));
		point.rgb = *reinterpret_cast<float*>(&rgb);
		point_cloud_ptr->points.push_back(point);
		
	}
	
	point_cloud_ptr->width = (int) point_cloud_ptr->points.size (); //number of points
	point_cloud_ptr->height = 1; //a list, one row of data
	
	cout << "Done" << endl;
	
	// write to file
	if (write_to_file) {
		//pcl::PLYWriter pw;
		//pw.write("pointcloud.ply",*mycloud);
		pcl::PCDWriter pw;
		pw.write("../ply_files/pointcloud.pcd",*point_cloud_ptr);
	}
}


//chirelity condition
bool SFM::testTriangulation(const Matx34d& P){
	vector<Point3f> pointcloud_projected(pointcloudXYZ.size());
	Matx44d P4x4 = Matx44d::eye();
	for(int i = 0; i<12; i++)
		P4x4.val[i] = P.val[i];
	perspectiveTransform(pointcloudXYZ, pointcloud_projected, P4x4);

	vector<uchar> status;
	status.resize(pointcloudXYZ.size(),0);
	for(int i =0; i<pointcloudXYZ.size(); i++)
		status[i] = (pointcloud_projected[i].z > 0) ? 1:0;

	int count = countNonZero(status);

	double percentage = ((double) count / (double) pointcloudXYZ.size());
	cout<<count<<"/"<<pointcloudXYZ.size()<<" = "<<percentage*100<<"% are infront of the camera"<<endl;
	if(percentage < 0.7)
		return false; //less than 80% of the points are in front of the camera
	//check for the coplanarity of points
	if(false){
		Mat_<double> cloud(pointcloudXYZ.size(),3);
		for(int i = 0; i<pointcloudXYZ.size();i++){
			cloud.row(i)(0) = pointcloudXYZ[i].x;
			cloud.row(i)(1) = pointcloudXYZ[i].y;
			cloud.row(i)(2) = pointcloudXYZ[i].z;
		}
		Mat_<double> mean;
		PCA pca(cloud, mean, CV_PCA_DATA_AS_ROW);

		int num_inliers = 0;
		Vec3d nrm = pca.eigenvectors.row(2);
		nrm = nrm/norm(nrm);
		Vec3f x0 = pca.mean;
		double p_to_plane_thresh = sqrt(pca.eigenvalues.at<double>(2));

		for(int i = 0; i<pointcloudXYZ.size();i++){
			Vec3f w = Vec3f(pointcloudXYZ[i]) - x0;
			double D = fabs(nrm.dot(w));
			if(D<p_to_plane_thresh)
				num_inliers++;
		}
		cout<<num_inliers<<"/"<<pointcloudXYZ.size()<<"are coplanar"<<endl;
		if((double) num_inliers / (double)(pointcloudXYZ.size())>0.85)
			return false;
	}
	return true;
}

Mat_<double> SFM::LinearLSTriangulation(Point3d u1, Matx34d P1, Point3d u2, Matx34d P2){
	//build matrix A for homogenous equation system Ax = 0
	//assume X = (x,y,z,1), for Linear-LS method
	//which turns it into a AX = B system, where A is 4x3, X is 3x1 and B is 4x1
	Matx43d A(u1.x*P1(2,0)-P1(0,0),	u1.x*P1(2,1)-P1(0,1),	u1.x*P1(2,2)-P1(0,2),		
			  u1.y*P1(2,0)-P1(1,0),	u1.y*P1(2,1)-P1(1,1),	u1.y*P1(2,2)-P1(1,2),		
			  u2.x*P2(2,0)-P2(0,0), u2.x*P2(2,1)-P2(0,1),	u2.x*P2(2,2)-P2(0,2),	
			  u2.y*P2(2,0)-P2(1,0), u2.y*P2(2,1)-P2(1,1),	u2.y*P2(2,2)-P2(1,2));
	Matx41d B(-(u1.x*P1(2,3)	-P1(0,3)),
			  -(u1.y*P1(2,3)	-P1(1,3)),
			  -(u2.x*P2(2,3)	-P2(0,3)),
			  -(u2.y*P2(2,3)	-P2(1,3)));
	Mat_<double> X;
	cv::solve(A,B,X,DECOMP_SVD);
	return X;
}

/**
 From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
 */
Mat_<double> SFM::IterativeLinearLSTriangulation(Point3d u1, Matx34d P1, Point3d u2, Matx34d P2){
	double wi1 = 1, wi2 = 1;
	Mat_<double> X(4,1);

	Mat_<double> X_temp = LinearLSTriangulation(u1,P1,u2,P2);
	X(0) = X_temp(0); X(1) = X_temp(1); X(2) = X_temp(2); X(3) = 1.0;

	//Hartley suggests 10 iterations are good enough
	for(int i =0; i<10; i++){
		//recalculate weights
		double p2x1 = Mat_<double>(Mat_<double>(P1).row(2)*X)(0);
		double p2x2 = Mat_<double>(Mat_<double>(P2).row(2)*X)(0);

		//break point
		if(fabsf(wi1 - p2x1) <= EPSILON && fabsf(wi2 - p2x2) <= EPSILON)
			break;
		wi1 = p2x1;
		wi2 = p2x2;

		//reqeight and solve
		Matx43d A((u1.x*P1(2,0)-P1(0,0))/wi1,	(u1.x*P1(2,1)-P1(0,1))/wi1,		(u1.x*P1(2,2)-P1(0,2))/wi1,		
				  (u1.y*P1(2,0)-P1(1,0))/wi1,	(u1.y*P1(2,1)-P1(1,1))/wi1,		(u1.y*P1(2,2)-P1(1,2))/wi1,		
				  (u2.x*P2(2,0)-P2(0,0))/wi2,	(u2.x*P2(2,1)-P2(0,1))/wi2,		(u2.x*P2(2,2)-P2(0,2))/wi2,	
				  (u2.y*P2(2,0)-P2(1,0))/wi2,	(u2.y*P2(2,1)-P2(1,1))/wi2,		(u2.y*P2(2,2)-P2(1,2))/wi2);
		Mat_<double> B = (Mat_<double>(4,1) <<	  -(u1.x*P1(2,3)	-P1(0,3))/wi1,
												  -(u1.y*P1(2,3)	-P1(1,3))/wi1,
												  -(u2.x*P2(2,3)	-P2(0,3))/wi2,
												  -(u2.y*P2(2,3)	-P2(1,3))/wi2);
		cv::solve(A,B,X_temp,DECOMP_SVD);
		X(0) = X_temp(0); X(1) = X_temp(1); X(2) = X_temp(2); X(3) = 1.0;
	}
	return X;
}

double SFM::triangulate(Matx34d P1, Matx34d P2){
	vector<KeyPoint> corresp;
	Matx44d P2_homo(P2(0,0),P2(0,1),P2(0,2),P2(0,3),
				    P2(1,0),P2(1,1),P2(1,2),P2(1,3),
				    P2(2,0),P2(2,1),P2(2,2),P2(2,3),
				    0,		0,		0,		1);
	cout<<"Triangulating"<<endl;
	//double t = getTickCount();
	std::vector<double> reproj_error;
	unsigned int pts_size = sel_points.size();

	////////////Method 1 (OpenCV triangulation)////////////////////////////
	Mat sel_points_undistorted,tracked_points_undistorted;
	undistortPoints(sel_points, sel_points_undistorted, K,D);
	undistortPoints(tracked_points, tracked_points_undistorted, K,D);
	//triangulate
	Mat sel_points_udrs = sel_points_undistorted.reshape(1, 2); //reshape
	Mat tracked_points_udrs = tracked_points_undistorted.reshape(1, 2); //reshape
	Mat points_3d(1,pts_size,CV_32FC4);
	cv::triangulatePoints(P1,P2,sel_points_udrs,tracked_points_udrs,points_3d);

	//calculate reprojection
	vector<Point3f> pts_3d;
	convertPointsHomogeneous(points_3d.reshape(4,1), pts_3d);
	Mat_<double> R = (cv::Mat_<double>(3,3) << P1(0,0),P1(0,1),P1(0,2), P1(1,0),P1(1,1),P1(1,2), P1(2,0),P1(2,1),P1(2,2));
	Vec3d rvec; Rodrigues(R ,rvec);
	Vec3d tvec(P1(0,3),P1(1,3),P1(2,3));
	vector<Point2f> reprojected_pt_set1;
	projectPoints(pts_3d,rvec,tvec,K,D,reprojected_pt_set1);
	pointcloudXYZ = pts_3d;
	for (int i=0; i<pts_size; i++) {		
		reproj_error.push_back(norm(sel_points[i]-reprojected_pt_set1[i]));
	}
	//cout<<pointcloudXYZ[0].x<<endl;
	////////////Method 2 (Custom triangulation)////////////////////////////
	/*Mat_<double> KP2 = K * Mat(P2);
	#pragma omp parallel for num_threads(1)
	for(int i= 0; i<pts_size; i++){
		Point2f kp1 = sel_points[i]; 
		Point3d u1(kp1.x,kp1.y,1.0);
		Mat_<double> um1 = K.inv() * Mat_<double>(u1); 
		u1.x = um1(0); u1.y = um1(1); u1.z = um1(2);

		Point2f kp2 = tracked_points[i];
		Point3d u2(kp2.x,kp2.y,1.0);
		Mat_<double> um2 = K.inv()*Mat_<double>(u2);
		u2.x = um2(0);
		u2.y = um2(1);
		u2.z = um2(2);

		Mat_<double> X = IterativeLinearLSTriangulation(u1,P1,u2,P2);
		//reproject
		Mat_<double> xpt_img = KP2 * X;
		Point2f xpt_img_temp(xpt_img(0)/xpt_img(2),xpt_img(1)/xpt_img(2));
		#pragma omp critical
		{
			pointcloudXYZ[i] = Point3f(X(0),X(1),X(2));
			reproj_error.push_back(norm(xpt_img_temp - kp2));

		}
	}*/
	Scalar mse = mean(reproj_error);
	cout << "Done. ("<<pointcloudXYZ.size()<<"points, " <<"s, mean reproj err = " << mse[0] << ")"<< endl;
	return mse[0];
}

bool SFM::getProjectionMatrix(){
	Mat k = K; //can be either K2
	E = k.t() * F * k;
	if(fabs(determinant(E)) > 1e-05){
		cout<<"Determinant of E != 0 : "<< determinant(E)<<endl;
		return false;
	}
	Mat_<double> R1(3,3), R2(3,3), t1(1,3),t2(1,3);
	Mat_<double> R(3,3), t(1,3);
	SVD svd1(E);
	Mat diag_matrix = (Mat_<double>(3,3) << 1, 0, 0, 0, 1, 0, 0, 0, 0);
	Mat w = (Mat_<double>(3,3) << 0, -1, 0, 1, 0, 0, 0, 0, 1);
	E = svd1.u*diag_matrix*(svd1.vt);

	//Using OpenCV Recoverpose
	double cx = k.at<double>(0,2);
	double cy = k.at<double>(1,2);
	double focal_length = k.at<double>(0,0);
	Point2d pp(cx,cy);
	Mat mask;
	recoverPose(E, sel_points, tracked_points, R, t, focal_length, pp, mask);
    Matx34d P1 = Matx34d(1,0,0,0,
	          			0,1,0,0,
			        	0,0,1,0);
	Matx34d P2 = Matx34d(R(0,0), R(0,1), R(0,2), t(0),
				         R(1,0), R(1,1), R(1,2), t(1),
				         R(2,0), R(2,1), R(2,2), t(2));
	
	double reproj_error1 = triangulate(P1,P2);
	double reproj_error2 = triangulate(P2,P1);

	if(!testTriangulation(P1)||!testTriangulation(P2)||reproj_error1 > 100.0 || reproj_error2 > 100.0){
		cout<<"The test is done - reproj_error1 = "<<reproj_error1<<"and reproj_error2 = "<<reproj_error2<<endl;
	}

	return true;
	//Custom
	/*SVD svd2(E);
	R1 = svd2.u*w*(svd2.vt); //Rotation solution 1
	R2 = svd2.u*w.t()*(svd2.vt); //Rotation solution 2
	if(determinant(R1) < 0)
		R1 = -R1;
	if(determinant(R2) < 0)
		R2 = -R2;
	t1 = svd2.u.col(2); //translation solution 1
	t2 = -svd2.u.col(2); //translation solution 2
	//cout<<"R1 : "<<R1<<endl<<"R2 :"<<R2<<endl<<"t1 :"<<t1<<endl<<"t2 :"<<t2<<endl;
	// Projection  Matrix
	Matx34d P1 = Matx34d(1,0,0,0,
	          			0,1,0,0,
			        	0,0,1,0);
	Matx34d P2 = Matx34d(R1(0,0),	R1(0,1),	R1(0,2),	t1(0),
				         R1(1,0),	R1(1,1),	R1(1,2),	t1(1),
				         R1(2,0),	R1(2,1),	R1(2,2),	t1(2));
	
	double reproj_error1 = triangulate(P1,P2);
	double reproj_error2 = triangulate(P2,P1);
	if(!testTriangulation(P1)||!testTriangulation(P2)||reproj_error1 > 100.0 || reproj_error2 > 100.0) {
		P2 = Matx34d(R1(0,0),	R1(0,1),	R1(0,2),	t2(0),
					 R1(1,0),	R1(1,1),	R1(1,2),	t2(1),
					 R1(2,0),	R1(2,1),	R1(2,2),	t2(2));
		cout<<"Testing P2"<<endl<<Mat(P2)<<endl;
		pointcloudXYZ.clear();
		reproj_error1 = triangulate(P1,P2);
	    reproj_error2 = triangulate(P2,P1);
	    if(!testTriangulation(P1)||!testTriangulation(P2)||reproj_error1 > 100.0 || reproj_error2 > 100.0){
	    	P2 = Matx34d(R2(0,0),	R2(0,1),	R2(0,2),	t2(0),
						 R2(1,0),	R2(1,1),	R2(1,2),	t2(1),
						 R2(2,0),	R2(2,1),	R2(2,2),	t2(2));
			cout << "Testing P2 "<< endl << Mat(P2) << endl;
			pointcloudXYZ.clear();
		    reproj_error1 = triangulate(P1,P2);
	    	reproj_error2 = triangulate(P2,P1);
	    	if(!testTriangulation(P1)||!testTriangulation(P2)||reproj_error1 > 100.0 || reproj_error2 > 100.0){
	    		cout << "Err is too big." << endl; 
				return false;
	    	}
	    }
	}*/
}

SFM::SFM(const char* calib_file_name){
	calib_file = FileStorage(calib_file_name, FileStorage::READ);
	calib_file["K"] >> K; //camera matrix
	calib_file["D1"] >> D; //Distortion matrix
}

void SFM::getFundamentalMatrix(){
	double min_val,max_val;
	minMaxIdx(sel_points,&min_val,&max_val);
	vector<uchar> status;
	F = findFundamentalMat(sel_points,tracked_points, FM_RANSAC,0.006*max_val,0.99, status );
	//F = findFundamentalMat(sel_points,tracked_points,CV_FM_8POINT);
	
	///Removing Outliers from the tracked points using Fundamental Matrix
	///To be used when computing Findamnetal matrix using RANSAC
	double status_nz = countNonZero(status);
	double status_sz = status.size();
	double kept_ratio = status_nz/status_sz;

	for(int i = 0; i<status.size();i++){
		Point2f pt = tracked_points.at(i);
		if(status.at(i) == 0 || pt.x<0 || pt.y<0){
			if(pt.x<0 || pt.y<0)
				status.at(i) = 0;
			sel_points.erase(sel_points.begin() + i);
			tracked_points.erase(tracked_points.begin() + i);
		}
	}
	//cout<<sel_points.size()<<" "<<tracked_points.size()<<endl;
}

void SFM::trackFeatures(Mat img_curr, Mat img_prev){
	vector<uchar> status;
	vector<float> err;
	Size winSize = Size(21,21);
	TermCriteria termcrit = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS,30,0.01);
	//Optical flow used to track points from previous frame to current frame
	calcOpticalFlowPyrLK(img_prev,img_curr,sel_points,tracked_points, status, err,winSize, 3, termcrit,0,0.001);

	//getting rid of the points for which the tracking failed
	//or points that are no longer in the frame
	int index_correction = 0;
	for(int i = 0; i<status.size();i++){
		Point2f pt = tracked_points.at(i-index_correction);
		if(status.at(i) == 0 || pt.x<0 || pt.y<0){
			if(pt.x<0 || pt.y<0)
				status.at(i) = 0;
			sel_points.erase(sel_points.begin() + (i-index_correction));
			tracked_points.erase(tracked_points.begin() + (i-index_correction));
		}
	}

}


void SFM::detectFeatures(Mat img){
	vector<cv::KeyPoint> key_points;
	int fast_threshold = 20;
	bool nonmaxSuppression = true;
	FAST(img,key_points,fast_threshold,nonmaxSuppression);
	KeyPoint::convert(key_points,sel_points,vector<int>());
}

void SFM::execute(vector<String> images_path){
	size_t count = images_path.size();
	//cout<<count<<endl;
	if(count < 2)
		cout<<"Insufficient images"<<endl;
	else{
		//Get the images from directory
		int i = 0;
		while(i<count){
			img = imread(images_path[i],1);
			//imshow("image",img);
			//waitKey(0);
			//Convert image to gray scale
			cvtColor(img, img_gray, COLOR_BGR2GRAY);
			if(img.empty()){
				cout<<"Empty image detected at frame number :"<<i<<endl;
				i++;
				continue;
			}
			if(i>=1){
				img_curr = img_gray;
				
				//Detect features from prev image
				detectFeatures(img_prev);
				
				//Track the detected features from the previous frame in current frame
				//to get matched points
				trackFeatures(img_prev,img_curr);
				
				//Get Fundamental matrix
				getFundamentalMatrix();

				//Get Essential Matrix
				bool result = getProjectionMatrix();
				cout<<sel_points[0]<<sel_points.size()<<" "<<tracked_points.size()<<" "<<pointcloudXYZ.size()<<endl;
				bool write_to_file = true;
				populatePCLPointCloud(pointcloudXYZ, write_to_file);
			}
			else{
				img_prev = img_gray;
			}
			//cout<<i<<endl;
			i++;
		}
	}
}