#include <cmath>     // exp, ceil
#include <cstdio>    // printf
#include <cstdlib>   // atof, atoi
#include <vector>    // std::vector
#include <omp.h>     // omp_get_wtime, omp_get_max_threads
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

double count = -1.0;
double total_px = 0;

int progress(){
    count = count + 1.0;
    //printf("\rPercent complete: %.2f%%", (count/total_px)*100);
    printf("\rPercent complete:  |%.*s%.*s|  %.2f%%", int(((count/total_px)*100)/10), "**********",int(10-((count/total_px)*100)/10), "----------", (count/total_px)*100);
    return 0;
}

int main(int argc, char** argv) {
    const char* input = "test_image.jpg";
    const char* output = "test_output.jpg";
    if (argc < 3) {
        std::printf("\n*Missing Arguments*\nusage: %s [input.jpg] [output.jpg] [sigma]\n\n**Test data will be used for this attempt.\n\n", argv[0]);
        //return 1;
    }else if (argc == 2){
        input =argv[1];
    }
    else{
        input =argv[1];
        output =argv[2];
    }
    bool devlog = false;
    if(argc >= 5) {
        if(std::string(argv[4]) == "--dev") devlog = true;
        
    }

    


    //* atof converts our string into a double-precision floating-point number. Otherwise default to 3
    double sigma = (argc >= 4) ? std::atof(argv[3]) : 5.0; 
    if (sigma <= 0.0) sigma = 1.0;

    int w, h, channels;
    //* load image
    unsigned char* img = stbi_load(input, &w, &h, &channels, 0);
    if (!img) {
        std::printf("could not load %s\n", input);
        return 1;
    }

    int radius = (int)std::ceil(3.0 * sigma); //? standard value based off sigma (can be modified if needed)
    int ksize  = 2 * radius + 1; //? size of kernel/map
    std::vector<double> kernel(ksize * ksize); 

    

    //* dy/dx are your offset from the center point/pixle. Hence why we start at '-radius'
    /*   -1 0 1
    \  1  x x x
    \  0  x o x             <-- example kernel. o is the center point/pixel/origin
    \ -1  x x x
    */
//! building the kernel serially. The nature of our radius is small so parallizing this will slow our program
//? rgblur ~~ raw gausian blur
    double sum = 0.0;
    for (int dy = -radius; dy <= radius; ++dy) 
        for (int dx = -radius; dx <= radius; ++dx) {
            double rgblur= std::exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma));
            kernel[(dy + radius) * ksize + (dx + radius)] =rgblur;
            sum +=rgblur;
        }
    
    for (double& rgblur: kernel) rgblur/= sum; //? normalizing rgblurvalues to sum to 1. This preserves the brightness of the image
    
    
    //* define output varaible with same size as original image
    std::vector<unsigned char> out(static_cast<size_t>(w) * h * channels); 


    double t0 = omp_get_wtime();
    if(devlog) {
        printf("echo -e \"\\e[?25h\" <- to get cursor back if program terminates\n");
        printf("Starting..");
        total_px = double(h);
        progress();
        fputs("\e[?25l", stdout); /* hide the cursor */
    }

    for (int y = 0; y < h; ++y) {
        
        for (int x = 0; x < w; ++x) {
            for (int c = 0; c < channels; ++c) {
                double acc = 0.0; //? var for accumulating the weight privately for each iteration

                //?blur for each pixle
                for (int dy = -radius; dy <= radius; ++dy) {
                int yy = y + dy;
                yy = (yy < 0) ? 0 : (yy >= h ? h - 1 : yy);
                for (int dx = -radius; dx <= radius; ++dx) {
                    int xx = x + dx;
                    xx = (xx < 0) ? 0 : (xx >= w ? w - 1 : xx);
                    acc += kernel[(dy + radius) * ksize + (dx + radius)]
                        * img[(static_cast<size_t>(yy) * w + xx) * channels + c];
                    }
                }
                out[(static_cast<size_t>(y) * w + x) * channels + c] = (unsigned char)(acc + 0.5);
            }
        }
        if (devlog) {
                progress();
        }
    }
    if (devlog) {
        fputs("\e[?25h", stdout); /* show the cursor */
        std::printf("\nWriting Image to jpg...\n");
    }
double t1 = omp_get_wtime();

//* write image and free memory after
stbi_write_jpg(output, w, h, channels, out.data(), 90);
stbi_image_free(img);

std::printf("blurred %dx%d (%d ch), sigma=%.2f, %d threads, %.3f s\n", w, h, channels, sigma, omp_get_max_threads(), t1 - t0);
return 0;
}