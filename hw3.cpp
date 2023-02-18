#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <sys/time.h>

using namespace std;

const int BLOCK_SIZE = 16384*16;

int elapse_time(timeval t1, timeval t2){
    return (t2.tv_sec - t1.tv_sec) * 1000000 + t2.tv_usec - t1.tv_usec;
}

void baseline(vector<int> &row_ptr, vector<int> &col_idx, vector<float> &value, vector<float> &B, vector<float> &C, int n){
    for(int i=0;i<n;++i){
        C[i] = 0.0;
        for(int j=row_ptr[i];j<row_ptr[i+1];j++){
            float col = col_idx[j];
            float val = value[j];
            C[i] += val*B[col];
        }
    }
}

int divide_csr_into_column_blocks(const std::vector<int> &row_ptr, const std::vector<int> &col_idx, const std::vector<float> &values, std::vector<std::vector<int> > &col_block_ptr) {
    int num_rows = row_ptr.size() - 1;
    int num_cols = 0;

    for (int i = 0; i < col_idx.size(); i++) {
        num_cols = std::max(num_cols, col_idx[i] + 1);
    }
    int num_col_blocks = (num_cols + BLOCK_SIZE - 1) / BLOCK_SIZE;

    col_block_ptr.resize(num_col_blocks);
    for (int i = 0; i < num_col_blocks; i++) {
        col_block_ptr[i].resize(num_rows + 1);
        col_block_ptr[i][0] = 0;
    }

    for (int i = 0; i < num_rows; i++) {
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
        int col = col_idx[j];
        int col_block = col / BLOCK_SIZE;
        col_block_ptr[col_block][i+1]++;
        }
    }

    for (int i = 0; i < num_col_blocks; i++) {
        for (int j = 1; j <= num_rows; j++) {
        col_block_ptr[i][j] += col_block_ptr[i][j - 1];
        }
        if(i+1 < num_col_blocks){
            col_block_ptr[i+1][0] += col_block_ptr[i][num_rows];
        }
    }
    return num_col_blocks;
}

void tiling(vector<vector<int> > &row_ptr, vector<int> &col_idx, vector<float> &value, vector<float> &B, vector<float> &C, int n, int nblocks){
    for(int b=0;b<nblocks;++b){
        for(int i=0;i<n;++i){
            for(int j=row_ptr[b][i]; j< row_ptr[b][i+1]; ++j){
                int col=col_idx[j];
                int val = value[j];
                C[i] += val * B[col];
            }  
        }
    }
}

void tiling_unroll(vector<vector<int> > &row_ptr, vector<int> &col_idx, vector<float> &value, vector<float> &B, vector<float> &C, int n, int nblocks){
    const int unroll_factor = 4; // unroll factor

    for(int b=0;b<nblocks;++b){
        for(int i=0;i<n;++i){
            int j=row_ptr[b][i];
            int j_end=row_ptr[b][i+1];
            while(j + unroll_factor <= j_end){
                int col0 = col_idx[j];
                int col1 = col_idx[j+1];
                int col2 = col_idx[j+2];
                int col3 = col_idx[j+3];

                int val0 = value[j];
                int val1 = value[j+1];
                int val2 = value[j+2];
                int val3 = value[j+3];

                C[i] += val0 * B[col0] + val1 * B[col1] + val2 * B[col2] + val3 * B[col3];
                j += unroll_factor;
            }
            for(; j<j_end; ++j){
                int col=col_idx[j];
                int val = value[j];
                C[i] += val * B[col];
            }  
        }
    }
}


int main()
{
    timeval base, tile, unroll, end;
    int row, col, m=0, n=0;
    vector<pair<int,int> > edges;
    vector<int> row_ptr, col_idx;
    vector<float> val, B, C, C2, C3;
    int loop_times = 20;

    // load data from file
    ifstream infile("../higgs-social_network.edgelist");
    while (infile >> row >> col) {
        edges.push_back(make_pair(row-1, col-1));
    }
    infile.close();

    // convert data list to csr format
    m =edges.size();
    for (const auto &edge : edges) {
        n = max(n, max(edge.first, edge.second) + 1);
    }
    row_ptr.resize(n+1);
    col_idx.resize(m);
    val.resize(m);

    fill(row_ptr.begin(), row_ptr.end(), 0);
    for (const auto &edge : edges) {
        row_ptr[edge.first + 1]++;
    }
    for (int i=0; i<n; i++) {
        row_ptr[i+1] += row_ptr[i];
    }
    for (int i = 0; i < m; i++) {
        col_idx[row_ptr[edges[i].first]++] = edges[i].second;
    }
    for (int i = n; i > 0; i--) {
        row_ptr[i] = row_ptr[i - 1];
    }
    row_ptr[0] = 0;
    
    for (int i = 0; i < m; i++) {
        val[i] = 1;
    }

    // initialize vector
    B.resize(n);
    for (int i = 0; i < n; i++) {
        B[i] = 1;
    }
    // initialize result
    C.resize(n);
    fill(C.begin(), C.end(), 0);
    C2.resize(n);
    fill(C2.begin(), C2.end(), 0);
    C3.resize(n);
    fill(C3.begin(), C3.end(), 0);

    cout << "Number of vertices: " << n << endl;
    cout << "Number of edges: " << m << endl;
    cout << endl;
    
    gettimeofday(&base, NULL);
    for(int i=0;i<loop_times;++i){
        baseline(row_ptr, col_idx, val, B, C, n);
    }
    gettimeofday(&tile, NULL);

    // for(int i=0;i<20;++i){
    //     cout<< C[i] << " ";
    // }
    // cout<<endl;

    std::vector<std::vector<int> > col_block_ptr;
    int nblocks = divide_csr_into_column_blocks(row_ptr, col_idx, val, col_block_ptr);
    
    gettimeofday(&tile, NULL);
    for(int i=0;i<loop_times;++i){
        tiling(col_block_ptr, col_idx, val, B, C2, n, nblocks);
    }
    gettimeofday(&unroll, NULL);

    // for(int i=0;i<20;++i){
    //     cout<< C2[i] << " ";
    // }
    // cout<<endl;
    
    gettimeofday(&unroll, NULL);
    for(int i=0;i<loop_times;++i){
        tiling_unroll(col_block_ptr, col_idx, val, B, C3, n, nblocks);
    }
    gettimeofday(&end, NULL);

    // for(int i=0;i<20;++i){
    //     cout<< C3[i] << " ";
    // }
    // cout<<endl;

    cout<< "baseline compute:" << elapse_time(base, tile) / loop_times <<endl;
    cout<< "tile compute:" << elapse_time(tile, unroll) / loop_times <<endl;
    cout<< "unroll compute:" << elapse_time(unroll, end) / loop_times <<endl;
    
    return 0;
}