
/*This version tries out nested forall/*
c program: Manelbrot1.c
--------------------------------
 1. draws Mandelbrot set for Fc(z)=z*z +c
 using Mandelbrot algorithm ( boolean escape time )
 2. OUTPUTS TO .pgm (portable grayscale file)
------------------------------- */

/* Interesting coordinatreL real: .26 and .27, imaginary part between 0 and .01
real : -.76 to -.74, imaginary .01 to .03, -1.26 to -1.24 , imaginary .01 to .03 */
#include <iostream>
#include <fstream>  // For file handling
#include <stdio.h>
#include <chrono>
#include <omp.h>
using namespace std; 
//#include <stdlib.h>
#include <math.h>

/*float CxMin = -0.76;*/
/*float CxMax = -0.74;*/
/*float CyMin = 0.01;*/
/*float CyMax= 0.03;*/

/*float CxMin = -1.26;*/
/*float CxMax = -1.24;*/
/*float CyMin = 0.01;*/
/*float CyMax= 0.03;*/

/* Full Mandelbrot set */
/*float CxMin = -2.0;
float CxMax =  1.0;
float CyMin = -1.25;
float CyMax= 1.75;*/

float CxMin = -1.27;
float CxMax = -1.24;
float CyMin = 0.01;
float CyMax= 0.03;
//#define output_row_width 20;
int output_row_width = 20;
/*#define iXmax 300;*/ /*image rows*/
/*#define iYmax 300;*/ /*image columns*/
/* bail-out value , radius of circle ;  */
/*#define EscapeRadius 2;*/
// #define EscapeRadius 2;
int EscapeRadius = 2;
#define IDim_max 700
//#define MaxGrayComponentValue 255;
int MaxGrayComponentValue = 255;
//#define IterationMax 1275;
int IterationMax = 1275;
float PixelWidth;
float PixelHeight;
int i,j;
float ER2;

int image_out[IDim_max+1][IDim_max+1];

void mandelrow(int my_i, int my_j, float my_Cy, int *pixel)
{
              float Zx, Zy;
              float Zx2, Zy2;
              float Cx;
              int Iteration;

              Cx= CxMin + my_j * PixelWidth;

              /* initial value of orbit = critical point Z= 0 */
              Zx =  0.0;
              Zy =  0.0;
              Zx2 = 0.0;
              Zy2 = 0.0;
              //for (Iteration = 0;Iteration <IterationMax && Zx2+Zy2< ER2;Iteration++)
              for (Iteration = 0; Iteration < IterationMax; Iteration++)
                {
                    if ((Zx2 + Zy2) >= ER2) {
                        break; // Exit the loop if the condition is met
                    }
                  Zy=2*Zx*Zy + my_Cy;
                  Zx=Zx2-Zy2 + Cx;
                  Zx2=Zx*Zx;
                  Zy2=Zy*Zy;
                };
            if (Iteration==IterationMax)
            /*  interior of Mandelbrot set = black Zero */
              *pixel= 0;
            else {
             *pixel= (int) Iteration/5;
              /**pixel= (int) Iteration;*/
            }
}

void write_image_to_file()
{
  int i, j;
  int OUTPUT_COUNTER = 1;
   // Create and open a file stream (output file)
    std::ofstream file("output.pgm");

    if (!file) {
        std::cerr << "Error opening file for writing." << std::endl;
    }

  file << "P2" << std::endl;
  file << IDim_max << " " << IDim_max << std::endl;
  file << MaxGrayComponentValue << std::endl;
        /*this loop write to the file in a manner acceptable to cstar 10 values per Row */
        for(i = 0; i < IDim_max; i++)
        {
          for ( j = 0; j < IDim_max; j++)
          {
            if (OUTPUT_COUNTER == output_row_width)
            {
              file << std::endl;
              OUTPUT_COUNTER = 1;
            }
            file << image_out[i][j] << " ";
            OUTPUT_COUNTER += 1;
          }
        }
    file.close();
 }
int main()
{
   #pragma omp parallel
    PixelWidth=(CxMax-CxMin)/IDim_max;
    PixelHeight=(CyMax-CyMin)/IDim_max;
    ER2=EscapeRadius*EscapeRadius;
    // Using chrono for high-resolution timing
    auto start = std::chrono::high_resolution_clock::now();
    //#pragma omp parallel for schedule(dynamid) //as opposed to static, as pixe calc time varies widely
     #pragma omp target teams distribute parallel for //trying this with T4GPU
  /* compute and write image data bytes to the file*/
      for ( i = 0; i < IDim_max; i++ )
      /*forall i = 0 to IDim_max - 1 grouping 25 do*/
       {
         float Cy;
         Cy = CyMin + i * PixelWidth;
         if (fabs(Cy)< PixelWidth/2)
            Cy=0.0;

         for( j = 0 ; j < IDim_max; j++) {

             mandelrow(i,j,Cy, &image_out[i][j]);
       }
     }
   auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "Time taken to calculate Mandelbrot set: " << duration.count() << " seconds" << std::endl;
 
  write_image_to_file();
  
}
