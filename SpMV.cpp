#include <cstdio>
#include <cassert>
#include <sys/time.h>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>

constexpr int BLOCK_SIZE = 3835826;

using namespace std;

class SpMV{
public:
    SpMV(){
        num_vertex = 0;
        num_edge = 0;
        num_block = 0;
        num_block = 0;
    };

    void ReadFile(string file_name);
    void ReBlocking();

    void MulBaseline();
    void MulTiling();
    void MulUnroll();
    
    
private:
    uint32_t num_vertex;
    uint32_t num_edge;
    uint32_t vaild_num_edge;
    uint32_t max_col_width;

    uint32_t num_block;

    vector<uint32_t> rowptr;
    vector<uint32_t> colidx;
    vector<uint32_t> data;
    vector<vector<uint32_t> > tiled_rowptr;
    vector<uint32_t> dense_vec;

    vector<uint32_t> res;
    vector<uint32_t> baseline_res;
    vector<uint32_t> tiling_res;
    vector<uint32_t> unroll_res;

};

void SpMV::ReadFile(string file_name){
    uint32_t v1, v2;
    vector<vector<uint32_t> > adj_list;

    ifstream ifs(file_name);
    while (ifs >> v1 >> v2) {
        if(v1>=adj_list.size()) adj_list.resize(v1+1);
        if(v2>=adj_list.size()) adj_list.resize(v2+1);
        adj_list[v1].push_back(v2);
        adj_list[v2].push_back(v1);
    }
    num_vertex = adj_list.size();
    for(auto& k:adj_list){
        sort(k.begin(), k.end());
        num_edge+=k.size();
    }
    printf("%u, %u\n", num_vertex, num_edge);
    rowptr.resize(num_vertex+1, 0);
    colidx.resize(num_edge, 0);
    data.resize(num_edge, 1);
    dense_vec.resize(num_vertex, 1);
    res.resize(num_vertex, 0);
    baseline_res.resize(num_vertex, 0);
    tiling_res.resize(num_vertex, 0);
    unroll_res.resize(num_vertex, 0);

    size_t cur = 0;
    for(size_t i=0; i<adj_list.size(); i++){
        for(size_t j=0; j<adj_list[i].size(); j++){
            colidx[cur++] = adj_list[i][j];
        }
        rowptr[i+1] = rowptr[i] + adj_list[i].size();
        assert(rowptr[i+1] == cur);
    }
}

void SpMV::ReBlocking(){
     max_col_width = 0;

    for (uint32_t i = 0; i < colidx.size(); i++) {
        max_col_width = std::max(max_col_width, colidx[i] + 1);
    }
    num_block = (max_col_width + BLOCK_SIZE - 1) / BLOCK_SIZE;

    tiled_rowptr.resize(num_block);
    for (uint32_t i = 0; i < num_block; i++) {
        tiled_rowptr[i].resize(num_vertex + 1);
        tiled_rowptr[i][0] = 0;
    }

    for (uint32_t i = 0; i < num_vertex; i++) {
        for (uint32_t j = rowptr[i]; j < rowptr[i + 1]; j++) {
            uint32_t block_id = colidx[j] / BLOCK_SIZE;
            tiled_rowptr[block_id][i+1]++;
        }
    }

    for (uint32_t i = 0; i < num_block; i++) {
        for (uint32_t j = 1; j <= num_vertex; j++) {
            tiled_rowptr[i][j] += tiled_rowptr[i][j - 1];
        }
        if(i+1 < num_block){
            tiled_rowptr[i+1][0] += tiled_rowptr[i][num_vertex];
        }
    }
    //return num_col_blocks;
}

void SpMV::MulBaseline(){
    for(uint32_t i=0;i<num_vertex;++i){
        baseline_res[i] = 0;
        for(uint32_t j=rowptr[i]; j<rowptr[i+1]; j++){
            uint32_t col = colidx[j];
            uint32_t val = data[j];
            baseline_res[i] += val*dense_vec[col];
        }
    }
    //int sum = accumulate(baseline_res.begin(), baseline_res.end(), 0);
    // printf("sum:%d\n", sum);
    // printf("MulBaseline finished\n");
}

void SpMV::MulTiling(){
    for(uint32_t b=0;b<num_block;++b){
        for(uint32_t i=0;i<max_col_width;++i){
            for(uint32_t j=tiled_rowptr[b][i]; j< tiled_rowptr[b][i+1]; ++j){
                uint32_t col = colidx[j];
                uint32_t val = data[j];
                tiling_res[i] += val * dense_vec[col];
            }  
        }
    }

    //int sum = accumulate(tiling_res.begin(), tiling_res.end(), 0);
    //printf("sum:%d\n", sum);
}

void SpMV::MulUnroll(){
    constexpr uint32_t unroll_size = 8; 

    for(uint32_t b=0;b<num_block;++b){
        for(uint32_t i=0;i<max_col_width;++i){
            uint32_t start=tiled_rowptr[b][i];
            uint32_t end=tiled_rowptr[b][i+1];
            uint32_t cur = start;
            while(cur + unroll_size <= end){
                uint32_t idx0 = colidx[cur];
                uint32_t idx1 = colidx[cur+1];
                uint32_t idx2 = colidx[cur+2];
                uint32_t idx3 = colidx[cur+3];
                uint32_t idx4 = colidx[cur+4];
                uint32_t idx5 = colidx[cur+5];
                uint32_t idx6 = colidx[cur+6];
                uint32_t idx7 = colidx[cur+7];

                uint32_t val0 = data[cur];
                uint32_t val1 = data[cur+1];
                uint32_t val2 = data[cur+2];
                uint32_t val3 = data[cur+3];
                uint32_t val4 = data[cur+4];
                uint32_t val5 = data[cur+5];
                uint32_t val6 = data[cur+6];
                uint32_t val7 = data[cur+7];

                unroll_res[i] +=    val0 * dense_vec[idx0] + 
                                    val1 * dense_vec[idx1] + 
                                    val2 * dense_vec[idx2] + 
                                    val3 * dense_vec[idx3] +
                                    val4 * dense_vec[idx4] + 
                                    val5 * dense_vec[idx5] + 
                                    val6 * dense_vec[idx6] + 
                                    val7 * dense_vec[idx7];
                cur += unroll_size;
            }
            while(cur<end){
                uint32_t idx = colidx[cur];
                uint32_t val = data[cur];
                unroll_res[i] += val * dense_vec[idx];
                ++cur;
            }  
        }
    }
    // int sum = accumulate(unroll_res.begin(), unroll_res.end(), 0);
    // printf("sum:%d\n", sum);
}


int ElapseTime(timeval start, timeval end){
    return (end.tv_sec - start.tv_sec) * 1000000 + end.tv_usec - start.tv_usec;
}

int main(int argc, char* argv[]){
    if(argc!=2){
        printf("Error! commandline should be: ./run.exe testfile\n");
        exit(-1);
    }

    SpMV spmv;
    spmv.ReadFile(argv[1]);

    timeval start, base, tile, unroll;

    spmv.ReBlocking();
    printf("ReBlocking finished\n");
    uint32_t test_times = 10;

    gettimeofday(&start, NULL);
    for(uint32_t i=0; i<test_times; i++){
        spmv.MulBaseline();
    }
    gettimeofday(&base, NULL);
    int elapse = ElapseTime(start, base)/test_times;
    printf("Baseline time:%d\n", elapse);


    gettimeofday(&start, NULL);
    for(uint32_t i=0; i<test_times; i++){
        spmv.MulTiling();
    }
    gettimeofday(&tile, NULL);
    elapse = ElapseTime(start, tile)/test_times;
    printf("MulTiling time:%d\n", elapse);

    gettimeofday(&start, NULL);
    for(uint32_t i=0; i<test_times; i++){
        spmv.MulUnroll();
    }
    gettimeofday(&unroll, NULL);
    elapse = ElapseTime(start, unroll)/test_times;
    printf("MulUnroll time:%d\n", elapse);

    
    return 0;
}
