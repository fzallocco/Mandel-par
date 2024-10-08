
/*This version tries out nested forall/*
c program: Manelbrot1.c
--------------------------------
 1. draws Mandelbrot set for Fc(z)=z*z +c
 using Mandelbrot algorithm ( boolean escape time )
 2. OUTPUTS TO .pgm (portable grayscale file)
------------------------------- */

/* Interesting coordinatreL real: .26 and .27, imaginary part between 0 and .01
real : -.76 to -.74, imaginary .01 to .03, -1.26 to -1.24 , imaginary .01 to .03 */
#include <stdlib.h>
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
#define output_row_width 20;
/*#define iXmax 300;*/ /*image rows*/
/*#define iYmax 300;*/ /*image columns*/
/* bail-out value , radius of circle ;  */
/*#define EscapeRadius 2;*/
#define EscapeRadius 2;
#define IDim_max 700
#define MaxGrayComponentValue 255;
#define IterationMax 1275;
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
              for (Iteration = 0;Iteration <IterationMax && ((Zx2+Zy2)< ER2);Iteration++)
                {

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
  cout << "P2" << ENDL;
  cout << IDim_max << " " << IDim_max << ENDL;
  cout << MaxGrayComponentValue << ENDL;
        /*this loop write to the file in a manner acceptable to cstar 10 values per Row */
        for(i = 0; i < IDim_max; i++)
        {
          for ( j = 0; j < IDim_max; j++)
          {
            if (OUTPUT_COUNTER == output_row_width)
            {
              cout << ENDL;
              OUTPUT_COUNTER = 1;
            }
            cout << image_out[i][j] << " ";
            OUTPUT_COUNTER += 1;
          }
        }
      }
int main()
{
    PixelWidth=(CxMax-CxMin)/IDim_max;
    PixelHeight=(CyMax-CyMin)/IDim_max;
    ER2=EscapeRadius*EscapeRadius;

  /* compute and write image data bytes to the file*/
      for ( i = 0; i < IDim_max; i++ )
      /*forall i = 0 to IDim_max - 1 grouping 25 do*/
      {
         float Cy;
         Cy = CyMin + i * PixelWidth;
         if (fabs(Cy)< PixelWidth/2)
            Cy=0.0;

         forall j = 0 to IDim_max - 1 grouping 25 do {

             mandelrow(i,j,Cy, &image_out[i][j]);
         }
      }
  write_image_to_file();
}
