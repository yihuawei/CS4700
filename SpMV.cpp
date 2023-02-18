#include <cstdio>
#include <cassert>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>

#define BLOCK_SIZE 32

using namespace std;


class SpMV{
public:
    SpMV(){
        num_vertex = 0;
        num_edge = 0;
    };

    void ReadFile(string file_name);
    void ReBlocking();

    void MulBaseline();
    void MulTiling();
    void MulUnroll();
    
    
private:
    uint32_t num_vertex;
    uint32_t num_edge;

    vector<uint32_t> rowptr;
    vector<uint32_t> colidx;
    vector<uint32_t> data;
    vector<vector<uint32_t> > tiled_rowptr;
    vector<uint32_t> dense_vec;

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
    
}

void SpMV::MulBaseline(){
    
}

void SpMV::MulTiling(){
    
}

void SpMV::MulUnroll(){
    
}



int main(int argc, char* argv[]){

    SpMV spmv;
    spmv.ReadFile("../higgs-social_network.edgelist");

    return 0;
}
