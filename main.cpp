#include <iostream>
#include <vector>
#include <memory.h>

using namespace std;
typedef int element;

vector<vector<element>> hybridmedianfilterN(vector<vector<element>> image, int MN_vecindad=1);
element median(element* elements, int N);

int main()
{
    cout << "Hello World!" << endl;
    vector<vector<int>> vect
            {
                {1, 2, 3, 100, 210, 40, 50},
                {4, 5, 6, 110, 230, 30, 55},
                {7, 8, 9, 140, 210, 20, 52},
                {4, 27, 36, 160, 20, 60, 50},
                {5, 58, 65, 10, 240, 70, 75},
                {7, 89, 94, 40, 220, 80, 82}
            };

    vect = hybridmedianfilterN(vect,1);
    cout << "Filas:"<<vect.size()<<"; Columnas:"<<vect[0].size() << endl;

    return 0;
}

//   MEDIAN calculation
//     elements - input elements
//     N        - number of input elements
element median(element* elements, int N)
{
   //   Order elements (only half of them)
   for (int i = 0; i < (N >> 1) + 1; ++i)
   {
      //   Find position of minimum element
      int min = i;
      for (int j = i + 1; j < N; ++j)
         if (elements[j] < elements[min])
            min = j;
      //   Put found minimum element in its place
      const element temp = elements[i];
      elements[i] = elements[min];
      elements[min] = temp;
   }
   //   Get result - the middle element
   return elements[N >> 1];
}


vector<vector<element>> hybridmedianfilterN(vector<vector<element>> image, int MN_vecindad)
{

    vector<vector<element>> img_result = image;
    //N width, M height
    element N = image[0].size();
    element M = image.size();
    //tamano de los elementos de la ventana
    int tW = (MN_vecindad*4) + 1;

    element window[tW];
    element results[3];

    int x,y,posicion;

   //   Move window through all elements of the image
   #pragma omp parallel for collapse(2) shared(img_result,image,M,N,MN_vecindad) private(window,results,x,y) schedule(static)
   for (y = MN_vecindad; y < M - MN_vecindad; ++y)
      for (x = MN_vecindad; x < N - MN_vecindad; ++x)
      {


         //   Pick up cross-window elements
         posicion=0;
         window[0] = image[y][x];
         for(int i=1; i<=MN_vecindad; i++){
             window[1+posicion] = image[(y-i)][x];
             window[2+posicion] = image[y][(x-i)];
             window[3+posicion] = image[y][(x+i)];
             window[4+posicion] = image[(y+i)][x];
             posicion=i*4;
         }
         //   Get median
         results[0] = median(window, tW);

         //   Pick up x-window elements
         window[0] = image[y][x];
         posicion=0;
         for(int i=1; i<=MN_vecindad; i++){
             window[1+posicion] = image[(y-i)][(x-i)];
             window[2+posicion] = image[(y-i)][(x+i)];
             window[3+posicion] = image[(y+i)][(x-i)];
             window[4+posicion] = image[(y+i)][(x+i)];
             posicion=i*4;
         }
         //   Get median
         results[1] = median(window, tW);
         //   Pick up leading element
         results[2] = image[y][x];
         //   Get result
         img_result[y][x] = median(results, 3);
      }


   return img_result;
}
