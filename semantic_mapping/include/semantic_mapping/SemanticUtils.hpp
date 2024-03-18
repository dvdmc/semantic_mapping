#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

static const std::vector<std::vector<uint8_t>> LABEL_TO_RGB_8 = {
    {0, 0, 0},                       // 0=background
    {128, 0, 128},                   // 5=bottle
    {192, 0, 0},                     // 9=chair
    {192, 128, 0},                   // 11=diningtable
    {192, 128, 128},                 // 15=person
    {0, 64, 0},                      // 16=pottedplant
    {0, 192, 0},                     // 18=sofa
    {0, 64, 128}};                   // 20=tvmonitor

static const std::vector<std::vector<uint8_t>> LABEL_TO_RGB_21 = {
    {0, 0, 0},                       // 0=background
    {0, 64, 0},                      // 1=aeroplane
    {0, 128, 0},                     // 2=bicycle
    {128, 128, 0},                   // 3=bird
    {0, 0, 128},                     // 4=boat
    {128, 0, 128},                   // 5=bottle
    {0, 128, 128},                   // 6=bus
    {128, 128, 128},                 // 7=car
    {64, 0, 0},                      // 8=cat
    {192, 0, 0},                     // 9=chair
    {64, 128, 0},                    // 10=cow
    {192, 128, 0},                   // 11=diningtable
    {64, 0, 128},                    // 12=dog
    {192, 0, 128},                   // 13=horse
    {64, 128, 128},                  // 14=motorbike
    {192, 128, 128},                 // 15=person
    {0, 64, 0},                      // 16=potted plant
    {128, 64, 0},                    // 17=sheep
    {0, 192, 0},                     // 18=sofa
    {128, 192, 0},                   // 19=train
    {0, 64, 128},                    // 20=tv/monitor
    {128, 64, 128}};                 // 21=background

static const std::vector<std::vector<uint8_t>> LABEL_TO_RGB_14 = {
    {0, 0, 0},       
    {255, 187, 120},                   // 1=bed
    {172, 114, 82},                    // 2=books
    {78, 71, 183},                     // 3=ceiling
    {188, 189, 34},                    // 4=chair
    {152, 223, 138},                   // 5=floor
    {140, 153, 101},                   // 6=furniture
    {255, 127, 14},                    // 7=objects
    {161, 171, 27},                    // 8=picture
    {190, 225, 64},                    // 9=sofa
    {206, 190, 59},                    // 10=table
    {115, 176, 195},                   // 11=tv
    {153, 108, 6},                     // 12=wall
    {247, 182, 210}};                  // 13=window

static const std::vector<uint8_t> interpolateColor(
    const std::vector<uint8_t> &color1, const std::vector<uint8_t> &color2) {
    std::vector<uint8_t> interpolatedColor;
    for (int i = 0; i < 3; i++) {
        interpolatedColor.push_back(
            (color1[i] + color2[i]) / 2);
    }
    return interpolatedColor;
}

static const std::vector<std::vector<uint8_t>> generateColorMap(int numLabels) {
    // Generic color map extracted from COCO labels
    const std::vector<std::vector<uint8_t>> genericMap = {
        {0, 0, 0},       {220, 20, 60},   {119, 11, 32},   {0, 0, 142},
        {0, 0, 230},     {106, 0, 228},   {0, 60, 100},    {0, 80, 100},
        {0, 0, 70},      {0, 0, 192},     {250, 170, 30},  {100, 170, 30},
        {220, 220, 0},   {175, 116, 175}, {250, 0, 30},    {165, 42, 42},
        {255, 77, 255},  {0, 226, 252},   {182, 182, 255}, {0, 82, 0},
        {120, 166, 157}, {110, 76, 0},    {174, 57, 255},  {199, 100, 0},
        {72, 0, 118},    {255, 179, 240}, {0, 125, 92},    {209, 0, 151},
        {188, 208, 182}, {0, 220, 176},   {255, 99, 164},  {92, 0, 73},
        {133, 129, 255}, {78, 180, 255},  {0, 228, 0},     {174, 255, 243},
        {45, 89, 255},   {134, 134, 103}, {145, 148, 174}, {255, 208, 186},
        {197, 226, 255}, {171, 134, 1},   {109, 63, 54},   {207, 138, 255},
        {151, 0, 95},    {9, 80, 61},     {84, 105, 51},   {74, 65, 105},
        {166, 196, 102}, {208, 195, 210}, {255, 109, 65},  {0, 143, 149},
        {179, 0, 194},   {209, 99, 106},  {5, 121, 0},     {227, 255, 205},
        {147, 186, 208}, {153, 69, 1},    {3, 95, 161},    {163, 255, 0},
        {119, 0, 170},   {0, 182, 199},   {0, 165, 120},   {183, 130, 88},
        {95, 32, 0},     {130, 114, 135}, {110, 129, 133}, {166, 74, 118},
        {219, 142, 185}, {79, 210, 114},  {178, 90, 62},   {65, 70, 15},
        {127, 167, 115}, {59, 105, 106},  {142, 108, 45},  {196, 172, 0},
        {95, 54, 80},    {128, 76, 255},  {201, 57, 1},    {246, 0, 122},
        {191, 162, 208}, {255, 255, 128}, {147, 211, 203}, {150, 100, 100},
        {168, 171, 172}, {146, 112, 198}, {210, 170, 100}, {92, 136, 89},
        {218, 88, 184},  {241, 129, 0},   {217, 17, 255},  {124, 74, 181},
        {70, 70, 70},    {255, 228, 255}, {154, 208, 0},   {193, 0, 92},
        {76, 91, 113},   {255, 180, 195}, {106, 154, 176}, {230, 150, 140},
        {60, 143, 255},  {128, 64, 128},  {92, 82, 55},    {254, 212, 124},
        {73, 77, 174},   {255, 160, 98},  {255, 255, 255}, {104, 84, 109},
        {169, 164, 131}, {225, 199, 255}, {137, 54, 74},   {135, 158, 223},
        {7, 246, 231},   {107, 255, 200}, {58, 41, 149},   {183, 121, 142},
        {255, 73, 97},   {107, 142, 35},  {190, 153, 153}, {146, 139, 141},
        {70, 130, 180},  {134, 199, 156}, {209, 226, 140}, {96, 36, 108},
        {96, 96, 96},    {64, 170, 64},   {152, 251, 152}, {208, 229, 228},
        {206, 186, 171}, {152, 161, 64},  {116, 112, 0},   {0, 114, 143},
        {102, 102, 156}, {250, 141, 255}
        };

    // Generate the color map add colors from the generic map. If the number of
    // labels is greater than the number of colors in the generic map, generate
    // new colors interpolating between colors.
    std::vector<std::vector<uint8_t>> colorMap;
    int repetitions_generic_map = std::ceil(numLabels / genericMap.size());
    for(int i = 0; i < numLabels; i++)
    {
        if(i < genericMap.size())
        {
            colorMap.push_back(genericMap[i]);
        }
        else
        {
            // Interpolate as many times as repetitions required
            std::vector<uint8_t> first_color = genericMap[i % genericMap.size()];
            std::vector<uint8_t> second_color = genericMap[(i + 1) % genericMap.size()];

            std::vector<uint8_t> interpolated_color = interpolateColor(first_color, second_color);
            for(int j = 0; j < repetitions_generic_map-1; j++)
            {
                interpolated_color = interpolateColor(interpolated_color, second_color);
            }
            colorMap.push_back(interpolated_color);
        }
    }

    return colorMap;
}

static const std::vector<std::vector<uint8_t>> getLabelMap(
    const int num_classes) {
    switch (num_classes) {
        case 8:
            return LABEL_TO_RGB_8;
        case 14:
            return LABEL_TO_RGB_14;
        case 21:
            return LABEL_TO_RGB_21;
        default:
            return generateColorMap(num_classes);
    }
}

static const std::vector<std::vector<uint8_t>> bgrToRgb(
    const std::vector<std::vector<uint8_t>> &bgrMap) {
    std::vector<std::vector<uint8_t>> rgbMap;
    for (const auto &color : bgrMap) {
        rgbMap.push_back({color[2], color[1], color[0]});
    }
    return rgbMap;
}

static const std::vector<std::vector<uint8_t>> rgbToBgr(
    const std::vector<std::vector<uint8_t>> &rgbMap) {
    std::vector<std::vector<uint8_t>> bgrMap;
    for (const auto &color : rgbMap) {
        bgrMap.push_back({color[2], color[1], color[0]});
    }
    return bgrMap;
}
