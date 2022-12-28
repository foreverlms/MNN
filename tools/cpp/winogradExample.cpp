//
//  winogradExample.cpp
//  MNN
//
//  Created by MNN on 2019/01/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "math/Matrix.hpp"
#include "math/WingoradGenerater.hpp"
#include <MNN/MNNDefine.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, const char *argv[])
{
    //    int unit       = ::atoi(argv[1]);
    //    int kernelSize = ::atoi(argv[2]);
    //    float interp   = 0.5f;
    //    if (argc > 3) {
    //        interp = ::atof(argv[3]);
    //    }
//    int unit = 2;
//    int kernelSize = 3;
    std::vector<int> unit{6,6}, kernelSize = {3,3};
    float interp = 1.0f;
    MNN::Math::WinogradGenerater generater(unit, kernelSize, interp, true);
    auto a = generater.A();
    auto b = generater.B();
    auto g = generater.G();

    auto BT = MNN::Math::Matrix::create(b->buffer().dim[0].extent, b->buffer().dim[1].extent);
    auto AT = MNN::Math::Matrix::create(a->buffer().dim[0].extent, a->buffer().dim[1].extent);
    MNN::Math::Matrix::transpose(BT,b.get());
    MNN::Math::Matrix::transpose(AT,a.get());

    MNN_PRINT("A=\n");
    MNN::Math::Matrix::print(AT );
    MNN_PRINT("G=\n");
    MNN::Math::Matrix::print(g.get());
    MNN_PRINT("B=\n");
    MNN::Math::Matrix::print(BT);
    return 0;
}
