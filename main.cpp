#include <iomanip>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <omp.h>

using namespace cv;
using namespace std;

Vec3b RandomColor(int value);

int main(int argc, char* argv[]){
    clock_t start,end;
    start=clock();

    Mat image=imread("/home/usuario/Descargas/imagen2.jpg");
    imshow("Imagen Original", image);

    Mat imageGray, imageCanny;
    cvtColor(image,imageGray,COLOR_BGR2GRAY);
    GaussianBlur(imageGray,imageGray,Size(5,5),2);
    Canny(imageGray,imageCanny,40,100);

    imshow("Imagen en gris", imageGray);
    imshow("Imagen Canny", imageCanny);

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    #pragma omp parallel
    findContours(imageCanny,contours,hierarchy,RETR_LIST,CHAIN_APPROX_SIMPLE, Point());
    Mat imageContours=Mat::zeros(image.size(),CV_8UC1);
    Mat marks(image.size(),CV_32S);
    marks=Scalar::all(0);
    int index=0;
    int compCount=0;
    for (;index>=0; index=hierarchy[index][0], compCount++)
    {
    drawContours(marks, contours, index, Scalar::all(compCount + 1), 1, 8, hierarchy);

    drawContours(imageContours, contours, index, Scalar(255), 1, 8, hierarchy);
    }
    Mat marksShows;
    convertScaleAbs(marks,marksShows);
    imshow("marksShows", marksShows);
    imshow("Imagen Contours",imageContours);
    watershed(image, marks);

    Mat afterWatershed;
    convertScaleAbs(marks,afterWatershed);
    imshow("After Watershed", afterWatershed);

    Mat PerspectiveImage = Mat::zeros(image.size(),CV_8UC3);
    #pragma omp parallel for
    for (int i=0; i< marks.rows; i++){
    for(int j=0; j<marks.cols; j++){
    int index=marks.at<int>(i,j);
    if(marks.at<int>(i,j)==-1){
    PerspectiveImage.at<Vec3b>(i,j)=Vec3b(255,255,255);
    }
    else{
    PerspectiveImage.at<Vec3b>(i,j)= RandomColor(index);
    }
    }
    }
    imshow("After ColorFill", PerspectiveImage);

    Mat wshed;
    addWeighted(image,0.4,PerspectiveImage, 0.6,0,wshed);
    imshow("AddWeighted Image", wshed);

    end=clock();
    double time_taken=double(end-start)/double(CLOCKS_PER_SEC);
    cout<<"El tiempo que ha tomado el programa es:" <<fixed << time_taken << setprecision(5);
    cout<<"segundos"<<endl;

    waitKey(0);
    return 0;
    }
    Vec3b RandomColor(int value){
    value=value%255;
    RNG rng;
    int aa=rng.uniform(0,value);
    int bb=rng.uniform(0,value);
    int cc=rng.uniform(0,value);
    return Vec3b(aa,bb,cc);
    }

