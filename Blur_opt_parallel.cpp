#include <cmath>     // exp, ceil
#include <cstdio>    // printf
#include <cstdlib>   // atof, atoi
#include <vector>    // std::vector
#include <omp.h>     // omp_get_wtime, omp_get_max_threads
#include <string>
#include <algorithm> // std::min, std::max

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

double count = -1.0;
double total_progress = 0;

int progress(bool x){
    count = count + 1.0;
    //printf("\rPercent complete: %.2f%%", (count/total_progress)*100);
    printf("\rPercent complete:  |%.*s%.*s|  %.2f%%", int(((count/total_progress)*100)/10), "**********",int(10-((count/total_progress)*100)/10), "----------", (count/total_progress)*100);
    if(!x) {
        printf("  -  1/2");
    } else {
        printf("  -  2/2");
    }

    printf("        ");
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
    
    if(argc >= 5 && std::string(argv[4]) == "--dev") {
        devlog = true;
        for (int i = 0; i < argc; i++) {
        printf("argv[%d] = %s\n", i, argv[i]);
        }
        printf("\nDEVLOG ACTIVE\n\n");
        
        printf("echo -e \"\\e[?25h\" <- to get cursor back if program terminates\n");
        printf("Starting...\n\n");
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
    
    if(devlog){
        total_progress = double(h);
        progress(false);
        fputs("\e[?25l", stdout); /* hide the cursor */
    }

    int radius = (int)std::ceil(3.0 * sigma); //? standard value based off sigma (can be modified if needed)
    int ksize  = 2 * radius + 1; //? size of kernel/map
    std::vector<double> kernel(ksize); 

    

    //* dy/dx are your offset from the center point/pixle. Hence why we start at '-radius'
    /*   -1 0 1
    \  1  x x x
    \  0  x o x             <-- example kernel. o is the center point/pixel/origin
    \ -1  x x x
    */
   //! building the kernel serially. The nature of our radius is small so parallizing this will slow our program
   //? rgblur ~~ raw gausian blur
    double sum = 0.0;
    for (int d = -radius; d <= radius; ++d) {
        double val = std::exp(-(d * d) / (2.0 * sigma * sigma));
        kernel[d + radius] = val;
        sum += val;
        }
    
    for (double& val: kernel) val /= sum; //? normalizing rgblurvalues to sum to 1. This preserves the brightness of the image
    
    
    //* define output varaible with same size as original image
    size_t img_bytes = static_cast<size_t>(w) * h * channels;
    std::vector<double> temp_buffer(img_bytes);
    std::vector<unsigned char> out(static_cast<size_t>(w) * h * channels); 

    int rows = h;

    double t0 = omp_get_wtime();
    //* parallize the loops. Run thread for each pixle staticly.
    

    #pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            size_t pixel_idx = (static_cast<size_t>(y) * w + x) * channels;
            for (int c = 0; c < channels; ++c) {
                double acc = 0.0; //? var for accumulating the weight privately for each iteration
                for (int dx = -radius; dx <= radius; ++dx) {
                    int xx = std::max(0, std::min(w - 1, x + dx));
                    size_t neighbor_idx = (static_cast<size_t>(y) * w + xx) * channels + c;
                    acc += kernel[dx + radius] * img[neighbor_idx];
                }
                temp_buffer[pixel_idx + c] = acc;    
            }
        }
        if (devlog) {
            #pragma omp critical
            progress(false);
        }
    }
    if(devlog) {
        count = 0;
        printf("\n\n");
    }
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            size_t pixel_idx = (static_cast<size_t>(y) * w + x) * channels;
            
            for (int c = 0; c < channels; ++c) {
                double acc = 0.0;
                for (int dy = -radius; dy <= radius; ++dy) {
                    int yy = std::max(0, std::min(h - 1, y + dy));
                    size_t neighbor_idx = (static_cast<size_t>(yy) * w + x) * channels + c;
                    acc += kernel[dy + radius] * temp_buffer[neighbor_idx];
                }
                out[pixel_idx + c] = static_cast<unsigned char>(acc + 0.5);
            }
        }
        if (devlog) {
            #pragma omp critical
            progress(true);
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