#include <iostream>
#include <cstdlib>
#include <vector>
#include <opencv2/opencv.hpp>

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
    
    if( !camera.open(cam_id) )
    {
      std::cout << "Error abriendo la camara" << std::endl;
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
    Rect r = faces[i];       
    rectangle(image2, Point(r.x+r.width, r.y+r.height), Point(r.x, r.y), Scalar(0, 255, 0), 2);
  }

  cv::imshow("faces", image2);
  
  key=cv::waitKey(5);
  if(key == 'q') break;

  }
  std::cout << "EXIT" << std::endl;
  destroyAllWindows();
}
