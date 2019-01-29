#include <iostream>
#include <cstdlib>
#include <vector>
#include <opencv2/opencv.hpp>
#include "cv.h"
#include "highgui.h"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

bool flag;
char key;
int main(int argc, char *argv[]) 
{
    cv::VideoCapture camera;

    cv::Mat image;
    cv::Mat image2;

    int cam_id=0;
    flag = true;
    Mat hat = imread("../img/hat.png", -1);
    Mat hat_resized;
    Mat moustache = imread("../img/moustache.png", -1);
    Mat moustache_resized;

    double color_pixel_0, color_pixel_1, color_pixel_2;

    if( !camera.open(cam_id) )
    {
      std::cout << "Error abriendo camara" << std::endl;
      return -1;
    }

    cv::CascadeClassifier face_cascade;
    face_cascade.load("../haarcascade_frontalface_default.xml");

  while (flag)
  {
    if(!camera.read(image))
    {
      std::cout << "No frame" << std::endl;
      cv::waitKey();
      flag = false;
    }

    cv::Mat gray;
    cv::cvtColor(image, gray, CV_BGR2GRAY);

    vector<Rect> faces;
    face_cascade.detectMultiScale(gray, faces, 1.3, 4, CASCADE_SCALE_IMAGE, Size(30, 30));

    image2 = image.clone();

    for (int i = 0; i < faces.size(); i++)
    {
      Rect face_i = faces[i];
      int facew = face_i.width;
      int faceh = face_i.height;
      Size hat_size(facew,faceh);
      resize(hat, hat_resized, hat_size);

      Size moustache_size(facew/2,faceh/2);
      resize(moustache, moustache_resized, moustache_size);

      double hat_locate = 0.50;
      double moustache_locate_y = 0.50;
      double moustache_move_x = (facew - moustache_resized.size[0])/2;

      for ( int j = 0; j < faceh ; j++)
      {
        for ( int k = 0; k < facew; k++)
        {

          double alpha_hat = hat_resized.at<cv::Vec4b>(j, k)[3] / 255.0;
          color_pixel_0 = (hat_resized.at<cv::Vec4b>(j, k)[0] * (alpha_hat)) + ((image2.at<cv::Vec3b>((face_i.y +j-(faceh*hat_locate)), (face_i.x +k))[0])* (1.0-alpha_hat));
          color_pixel_1 = (hat_resized.at<cv::Vec4b>(j, k)[1] * (alpha_hat)) + ((image2.at<cv::Vec3b>((face_i.y +j-(faceh*hat_locate)), (face_i.x +k))[1])* (1.0-alpha_hat));
          color_pixel_2 = (hat_resized.at<cv::Vec4b>(j, k)[2] * (alpha_hat)) + ((image2.at<cv::Vec3b>((face_i.y +j-(faceh*hat_locate)), (face_i.x +k))[2])* (1.0-alpha_hat));

          if((face_i.y +j-(faceh*hat_locate))>0)
          {
            image2.at<cv::Vec3b>((face_i.y +j-(faceh*hat_locate)), (face_i.x +k))[0] = color_pixel_0 ;
            image2.at<cv::Vec3b>((face_i.y +j-(faceh*hat_locate)), (face_i.x +k))[1] = color_pixel_1 ;
            image2.at<cv::Vec3b>((face_i.y +j-(faceh*hat_locate)), (face_i.x +k))[2] = color_pixel_2 ;
          }

          if((j<(faceh/2))&&(k<(facew/2)))
          {
            double alpha_moustache = moustache_resized.at<cv::Vec4b>(j, k)[3] / 255.0;
            color_pixel_0 = (moustache_resized.at<cv::Vec4b>(j, k)[0] * (alpha_moustache)) + ((image2.at<cv::Vec3b>((face_i.y +j+(faceh*moustache_locate_y)), (face_i.x +k+(moustache_move_x)))[0])* (1.0-alpha_moustache));
            color_pixel_1 = (moustache_resized.at<cv::Vec4b>(j, k)[1] * (alpha_moustache)) + ((image2.at<cv::Vec3b>((face_i.y +j+(faceh*moustache_locate_y)), (face_i.x +k+(moustache_move_x)))[1])* (1.0-alpha_moustache));
            color_pixel_2 = (moustache_resized.at<cv::Vec4b>(j, k)[2] * (alpha_moustache)) + ((image2.at<cv::Vec3b>((face_i.y +j+(faceh*moustache_locate_y)), (face_i.x +k+(moustache_move_x)))[2])* (1.0-alpha_moustache));

            image2.at<cv::Vec3b>((face_i.y +j+(faceh*moustache_locate_y)), (face_i.x +k+(moustache_move_x)))[0] = color_pixel_0 ;
            image2.at<cv::Vec3b>((face_i.y +j+(faceh*moustache_locate_y)), (face_i.x +k+(moustache_move_x)))[1] = color_pixel_1 ;
            image2.at<cv::Vec3b>((face_i.y +j+(faceh*moustache_locate_y)), (face_i.x +k+(moustache_move_x)))[2] = color_pixel_2 ;
          }
        }
      }
    }

    cv::imshow("faces", image2);

    key=cv::waitKey(5);
    if(key == 'q') break;
    }

    destroyAllWindows();
  }
