#include <torch/script.h> // One-stop header.
#include <iostream>
#include <memory>
#include "opencv2/imgproc/imgproc_c.h"//cvload
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <time.h>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <string>

using namespace cv;
using namespace std;
vector<float> box_pr_final;
vector<float> x_pos;
vector<float> y_pos;
int main(int argc, const char* argv[]) {
	clock_t start = clock();
	/*if (argc != 2) {
		std::cerr << "usage: example-app <path-to-exported-script-module>\n";
		return -1;
	}*/

	std::string path = "object_model480640.pt";
	//std::cout << path << std::endl;

	torch::jit::script::Module module;
	module = torch::jit::load(path);
	module.to(at::kCUDA);
	//module = torch::jit::load(path);
	//try {
	//	 //Deserialize the ScriptModule from a file using torch::jit::load().
	//	module = torch::jit::load(path);
	//	module.to(at::kCUDA);
	//	std::cout <<6666 << std::endl;

	//}
	//catch (const c10::Error & e) {
	//	std::cerr << "error loading the model\n";
	//	auto m = e.what();
	//	std::cerr << m;
	//	return -1;
	//}
	clock_t ends = clock();
	cv::Mat image;
	int h = 15;
	int w = 20;
	int batch = 1;
	int anchor_dim = 1;
	int num_classes = 1;
	bool only_objectness = true;
	image = cv::imread("320.jpg");
	
	
	torch::Tensor img_tensor = torch::from_blob(image.data, { 1,image.rows, image.cols, 3 }, torch::kByte);
	img_tensor = img_tensor.permute({ 0,3,1,2 });
	img_tensor = img_tensor.toType(torch::kFloat32);
	img_tensor = img_tensor.to(torch::kCUDA);
	img_tensor = img_tensor.div(255);
	try {
		auto output0 = module.forward({ img_tensor }).toTensor();
		//auto output = output0.toTensor();
	}
	catch (const std::runtime_error & e) {
		
		auto m = e.what();
		std::cerr << m;
		return -1;
	}
	clock_t ends3 = clock();
	auto output = module.forward({ img_tensor }).toTensor();
	torch::Tensor result = output.view({ 1, 20, 300 });
	result = result.transpose(0, 1).contiguous();
	result = result.view({ 20,300 });
	torch::Tensor grid_x = torch::arange(0, w, 1).repeat({ h,1 }).repeat({ batch * anchor_dim, 1, 1 }).view({ batch * anchor_dim * h * w }).cuda();
	torch::Tensor grid_y = torch::arange(0, h, 1).repeat({ w,1 }).t().repeat({ batch * anchor_dim, 1, 1 }).view({ batch * anchor_dim * h * w }).cuda();
	torch::Tensor xs0 = result[0].sigmoid_() + grid_x;
	torch::Tensor ys0 = result[1].sigmoid_() + grid_y;
	torch::Tensor xs1 = result[2] + grid_x;
	torch::Tensor ys1 = result[3] + grid_y;
	torch::Tensor xs2 = result[4] + grid_x;
	torch::Tensor ys2 = result[5] + grid_y;
	torch::Tensor xs3 = result[6] + grid_x;
	torch::Tensor ys3 = result[7] + grid_y;
	torch::Tensor xs4 = result[8] + grid_x;
	torch::Tensor ys4 = result[9] + grid_y;
	torch::Tensor xs5 = result[10] + grid_x;
	torch::Tensor ys5 = result[11] + grid_y;
	torch::Tensor xs6 = result[12] + grid_x;
	torch::Tensor ys6 = result[13] + grid_y;
	torch::Tensor xs7 = result[14] + grid_x;
	torch::Tensor ys7 = result[15] + grid_y;
	torch::Tensor xs8 = result[16] + grid_x;
	torch::Tensor ys8 = result[17] + grid_y;
	torch::Tensor det_confs = result[18].sigmoid_();

	at::Tensor x = result[19];
	torch::Tensor cls_confs = torch::softmax(result[19].view({ 300,1 }), 1);
	//std::cout <<cls_confs<< '\n';
	std::tuple<torch::Tensor, torch::Tensor>  cls_max = torch::max(cls_confs, 1);
	// class score
	auto cls_max_confs = std::get<0>(cls_max);
	// index
	auto cls_max_ids = std::get<1>(cls_max);
	clock_t ends2 = clock();

	
	int sz_hw;
	int sz_hwa;
	sz_hw = h * w;
	sz_hwa = sz_hw;
	det_confs = det_confs.cpu();
	xs0 = xs0.cpu();
	ys0 = ys0.cpu();
	xs1 = xs1.cpu();
	ys1 = ys1.cpu();
	xs2 = xs2.cpu();
	ys2 = ys2.cpu();
	xs3 = xs3.cpu();
	ys3 = ys3.cpu();
	xs4 = xs4.cpu();
	ys4 = ys4.cpu();
	xs5 = xs5.cpu();
	ys5 = ys5.cpu();
	xs6 = xs6.cpu();
	ys6 = ys6.cpu();
	xs7 = xs7.cpu();
	ys7 = ys7.cpu();
	xs8 = xs8.cpu();
	ys8 = ys8.cpu();

	int max_conf = -1;
	int ind;
	vector<float> box;
	vector<vector<float>> boxes;
	
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			ind = i * w + j;
			torch::Tensor det_conf = det_confs[ind];
			if (only_objectness)
			{
				torch::Tensor conf = det_confs[ind];
				//std::cout<< conf.item<float>()<<'\n';
				if (conf.item<float>() > 0.1)
				{
					torch::Tensor bcx0 = xs0[ind];
					torch::Tensor bcy0 = ys0[ind];
					torch::Tensor bcx1 = xs1[ind];
					torch::Tensor bcy1 = ys1[ind];
					torch::Tensor bcx2 = xs2[ind];
					torch::Tensor bcy2 = ys2[ind];
					torch::Tensor bcx3 = xs3[ind];
					torch::Tensor bcy3 = ys3[ind];
					torch::Tensor bcx4 = xs4[ind];
					torch::Tensor bcy4 = ys4[ind];
					torch::Tensor bcx5 = xs5[ind];
					torch::Tensor bcy5 = ys5[ind];
					torch::Tensor bcx6 = xs6[ind];
					torch::Tensor bcy6 = ys6[ind];
					torch::Tensor bcx7 = xs7[ind];
					torch::Tensor bcy7 = ys7[ind];
					torch::Tensor bcx8 = xs8[ind];
					torch::Tensor bcy8 = ys8[ind];
					torch::Tensor cls_max_conf = cls_max_confs[ind];
					torch::Tensor cls_max_id = cls_max_ids[ind];
					box.push_back(bcx0.item<float>() / w);
					box.push_back(bcy0.item<float>() / h);
					box.push_back(bcx1.item<float>() / w);
					box.push_back(bcy1.item<float>() / h);
					box.push_back(bcx2.item<float>() / w);
					box.push_back(bcy2.item<float>() / h);
					box.push_back(bcx3.item<float>() / w);
					box.push_back(bcy3.item<float>() / h);
					box.push_back(bcx4.item<float>() / w);
					box.push_back(bcy4.item<float>() / h);
					box.push_back(bcx5.item<float>() / w);
					box.push_back(bcy5.item<float>() / h);
					box.push_back(bcx6.item<float>() / w);
					box.push_back(bcy6.item<float>() / h);
					box.push_back(bcx7.item<float>() / w);
					box.push_back(bcy7.item<float>() / h);
					box.push_back(bcx8.item<float>() / w);
					box.push_back(bcy8.item<float>() / h);
					box.push_back(det_conf.item<float>());
					box.push_back(cls_max_conf.item<float>());
					box.push_back(cls_max_id.item<float>());
					boxes.push_back(box);
					box.clear();


				}

			}

		}
	}
	float best_conf_est = 0;
	vector<float> box_pr;
	for (int boxid = 0; boxid < boxes.size(); boxid++)
	{

		if (boxes[boxid][18] > best_conf_est)
		{
			box_pr = boxes[boxid];		
			best_conf_est = boxes[boxid][18];
		}
	}
	// std::cout<<"dd"<<box_pr<<'\n';
	//vector<float> box_pr_final;
	//vector<float> x_pos;
	//<float> y_pos;

	for (int i = 0; i < 18; i++)
	{
		if (i % 2 == 0) {
			float x = box_pr[i] * 640.0 * 1280/640.0;
			box_pr_final.push_back(x);
			x_pos.push_back(x);
		}
		else
		{
			float y = box_pr[i]*480*1024.0/480;
			box_pr_final.push_back(y);
			y_pos.push_back(y);

		}
	}
	clock_t ends1 = clock();
	cout << "Loadmodel Time : " << (double)(ends - start) / CLOCKS_PER_SEC << endl;
	cout << "Compute Time : " << (double)(ends1 - ends) / CLOCKS_PER_SEC << endl;
	cout << "Compute Time : " << (double)(ends2 - ends3) / CLOCKS_PER_SEC << endl;

	// std::cout<<"dd"<<box_pr_final<<'\n';
	// std::cout<<"dd"<<x_pos<<'\n';
	// std::cout<<"dd"<<y_pos<<'\n';
	cv::Mat image1;
	image1 = cv::imread("D:\\g00\\singleshotpose\\LINEMOD\\object\\JPEGImages\\320.jpg");
	//IplImage* img = cv::cvLoadImage("/home/lab606/libtorch/example-app/image/304.jpg");
	for (int i = 0; i < x_pos.size(); i++)
	{
		circle(image1, Point((int)x_pos[i], (int)y_pos[i]), 2, (255, 255, 0), 1);
	}
	line(image1, cvPoint((int)x_pos[1], (int)y_pos[1]), cvPoint((int)x_pos[2], (int)y_pos[2]), (255, 255, 255), 1, 8);
	line(image1, cvPoint((int)x_pos[1], (int)y_pos[1]), cvPoint((int)x_pos[3], (int)y_pos[3]), (255, 255, 255), 1, 8);
	line(image1, cvPoint((int)x_pos[1], (int)y_pos[1]), cvPoint((int)x_pos[5], (int)y_pos[5]), (255, 255, 255), 1, 8);
	line(image1, cvPoint((int)x_pos[6], (int)y_pos[6]), cvPoint((int)x_pos[2], (int)y_pos[2]), (255, 255, 255), 1, 8);
	line(image1, cvPoint((int)x_pos[2], (int)y_pos[2]), cvPoint((int)x_pos[4], (int)y_pos[4]), (255, 255, 255), 1, 8);
	line(image1, cvPoint((int)x_pos[3], (int)y_pos[3]), cvPoint((int)x_pos[4], (int)y_pos[4]), (255, 255, 255), 1, 8);
	line(image1, cvPoint((int)x_pos[3], (int)y_pos[3]), cvPoint((int)x_pos[7], (int)y_pos[7]), (255, 255, 255), 1, 8);
	line(image1, cvPoint((int)x_pos[4], (int)y_pos[4]), cvPoint((int)x_pos[8], (int)y_pos[8]), (255, 255, 255), 1, 8);
	line(image1, cvPoint((int)x_pos[5], (int)y_pos[5]), cvPoint((int)x_pos[6], (int)y_pos[6]), (255, 255, 255), 1, 8);
	line(image1, cvPoint((int)x_pos[5], (int)y_pos[5]), cvPoint((int)x_pos[7], (int)y_pos[7]), (255, 255, 255), 1, 8);
	line(image1, cvPoint((int)x_pos[6], (int)y_pos[6]), cvPoint((int)x_pos[8], (int)y_pos[8]), (255, 255, 255), 1, 8);
	line(image1, cvPoint((int)x_pos[7], (int)y_pos[7]), cvPoint((int)x_pos[8], (int)y_pos[8]), (255, 255, 255), 1, 8);
	//cvNamedWindow("src");
	imshow("image1", image1);
	waitKey();
	 
	 std::vector<cv::Point2f> Generate2DPoints();
	 std::vector<cv::Point3f> Generate3DPoints();

	 std::vector<cv::Point2f> imagePoints = Generate2DPoints();
	 std::vector<cv::Point3f> objectPoints = Generate3DPoints();

	 std::cout << "There are " << imagePoints.size() << " imagePoints and " << objectPoints.size() << " objectPoints." << std::endl;
	 
	 double fx = 1055.1268f;; //focal length x
	 double fy = 1055.5656f; //focal le

	 double cx = 655.9771f;//optical centre x
	 double cy = 522.0734f; //optical centre y

	 cv::Mat cameraMatrix(3, 3, cv::DataType<double>::type);
	 cameraMatrix.at<double>(0, 0) = fx;
	 cameraMatrix.at<double>(1, 1) = fy;
	 cameraMatrix.at<double>(2, 2) = 1;
	 cameraMatrix.at<double>(0, 2) = cx;
	 cameraMatrix.at<double>(1, 2) = cy;
	 cameraMatrix.at<double>(0, 1) = 0;
	 cameraMatrix.at<double>(1, 0) = 0;
	 cameraMatrix.at<double>(2, 0) = 0;
	 cameraMatrix.at<double>(2, 1) = 0;


	 std::cout << "Initial cameraMatrix: " << cameraMatrix << std::endl;

	 cv::Mat distCoeffs(4, 1, cv::DataType<double>::type);
	 distCoeffs.at<double>(0) = 0;
	 distCoeffs.at<double>(1) = 0;
	 distCoeffs.at<double>(2) = 0;
	 distCoeffs.at<double>(3) = 0;

	 cv::Mat rvec(3, 1, cv::DataType<double>::type);
	 cv::Mat tvec(3, 1, cv::DataType<double>::type);

	 cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);

	 std::cout << "rvec: " << rvec << std::endl;
	 std::cout << "tvec: " << tvec << std::endl;

	 std::vector<cv::Point2f> projectedPoints;
	 cv::projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);

	 for (unsigned int i = 0; i < projectedPoints.size(); ++i)
	 {
		 std::cout << "Image point: " << imagePoints[i] << " Projected to " << projectedPoints[i] << std::endl;
	 }

	 return 0;


}

std::vector<cv::Point2f> Generate2DPoints()
{
	std::vector<cv::Point2f> points;

	//points.push_back(cv::Point2f(x_pos[0], y_pos[0]));

	points.push_back(cv::Point2f(x_pos[1], y_pos[1]));

	points.push_back(cv::Point2f(x_pos[2], y_pos[2]));

	points.push_back(cv::Point2f(x_pos[3], y_pos[3]));

	points.push_back(cv::Point2f(x_pos[4], y_pos[4]));

	points.push_back(cv::Point2f(x_pos[5], y_pos[5]));

	points.push_back(cv::Point2f(x_pos[6], y_pos[6]));
	points.push_back(cv::Point2f(x_pos[7], y_pos[7]));

	points.push_back(cv::Point2f(x_pos[8], y_pos[8]));

	//points.push_back(cv::Point2f(x_pos[9], y_pos[9]));



	for (unsigned int i = 0; i < points.size(); ++i)
	{
		std::cout << points[i] << std::endl;
	}

	return points;
}

std::vector<cv::Point3f> Generate3DPoints()
{
	std::vector<cv::Point3f> points;
	float x, y, z;
	//x = 0; y = 0; z = 0;
	//points.push_back(cv::Point3f(x, y, z));

	x = -0.059568; y = -0.051870; z = -0.037123;
	points.push_back(cv::Point3f(x * 1000, y * 1000, z * 1000));
	x = -0.059568; y = -0.051870; z = 0.037123;
	points.push_back(cv::Point3f(x * 1000, y * 1000, z * 1000));

	x = -0.059568; y = 0.051870; z = -0.037123;
	points.push_back(cv::Point3f(x * 1000, y * 1000, z * 1000));

	x = -0.059568; y = 0.051870; z = 0.0371230;
	points.push_back(cv::Point3f(x * 1000, y * 1000, z * 1000));

	x = 0.059568; y = -0.051870; z = -0.037123;
	points.push_back(cv::Point3f(x * 1000, y * 1000, z * 1000));

	x = 0.059568; y = -0.051870; z = 0.037123;
	points.push_back(cv::Point3f(x * 1000, y * 1000, z * 1000));
	x = 0.059568; y = 0.051870; z = -0.037123;
	points.push_back(cv::Point3f(x * 1000, y * 1000, z * 1000));
	x = 0.059568; y = 0.051870; z = 0.037123;
	points.push_back(cv::Point3f(x * 1000, y * 1000, z * 1000));
	for (unsigned int i = 0; i < points.size(); ++i)
	{
		std::cout << points[i] << std::endl;
	}

	return points;
}
