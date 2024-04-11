

#ifndef DYNAMIC_DCORE_HINDEX_H
#define DYNAMIC_DCORE_HINDEX_H

#include "Common.h"
class hIndex final{
private:
    typedef  struct final{
        uint32_t vid;
        uint32_t eid;
    } ArrayEntry;
    // data members
    uint32_t n_;  // the # of vertices
    uint32_t m_;  // the # of edges

    //uint32_t M_;  // the smaller k_max value of inserted/deleted edge before graph modification
    //uint32_t N_;  // the bigger k_max value of inserted/deleted edge after graph modification

    // the adjacency array representation
    vector<vector<ArrayEntry>> adj_in;
    vector<vector<ArrayEntry>> adj_out;
    //vector<vector<ArrayEntry>> M_adj_in, M_adj_out; //adj of (M_, 0)-core
    //vector<vector<ArrayEntry>> M_plus_1_adj_in, M_plus_1_adj_out; //adj of (M_+1, 0)-core / {(M_-1, 0)-core for edge deletion }

    // the set of edges
    vector<pair<uint32_t, uint32_t>> edges_;
    vector<uint32_t> nodes_;

    // the set of vertices have their kmax changed after Kmax value maintenance
    //vector<uint32_t> dif_kmax_M_group;


    uint32_t cal_hIndex (const vector<uint32_t> &input_vector);
    void kMaxRemove (vector<uint32_t> &m_d, vector<bool> &vbDeleted, uint32_t cur_node, const uint32_t &M);
    void kMaxFindIncore(uint32_t root_node_id, vector<uint32_t> &m_d, vector<vector<ArrayEntry>> &sub_adj_in,
                        vector<vector<ArrayEntry>> &sub_adj_out, vector<bool> &be_in_incore, const uint32_t &M_);
    void lMaxFindOutcore(uint32_t root_node_id, vector<uint32_t> &m_d, vector<vector<ArrayEntry>> &sub_adj_in,
                         vector<vector<ArrayEntry>> &sub_adj_out, vector<bool> &be_in_outcore,
                         uint32_t  &k,uint32_t &k_M_, vector<vector<ArrayEntry>> &k_adj_out);

public:
    hIndex(vector<pair<uint32_t, uint32_t> > & edges, const  vector<vector<pair<::uint32_t,::uint32_t>>> & old_d_core_decomposition);   //onetime initialization
    hIndex(vector<pair<uint32_t, uint32_t> > & edges, const  vector<vector<pair<::uint32_t,::uint32_t>>> & old_d_core_decomposition, vector<pair<uint32_t, uint32_t>> & modified_edges);   //multiple initialization
    hIndex(const hIndex&) = delete;
    hIndex& operator=(const hIndex&) = delete;

    //vector<::uint32_t> in_coreness;


    void maintainKmax(const vector<pair<uint32_t, uint32_t>> & modified_edges, bool is_insert, const int & lmax_number_of_threads);  // maintain the kmax value of vertices, is_insert = true means insertion, otherwise deletion
    void maintainKmaxDfs(const vector<pair<uint32_t, uint32_t>> & modified_edges, bool is_insert, const uint32_t &M);  //for single edge updates, maintain the kmax value of vertices
    void maintainKmaxSingle(const vector<pair<uint32_t, uint32_t>> & modified_edges, bool is_insert, const int & lmax_number_of_threads);  //for single edge updates, maintain the kmax value of vertices with Hindex,
    void maintainKlist(const vector<pair<uint32_t, uint32_t>> & modified_edges, bool is_insert, const uint32_t &M,
                       bool k0core_pruning, bool reuse_pruning ,const int &lmax_number_of_threads);  // based on the updated kmax value, maintain the k-lists, i.e., l_{max}(v,k) value
    void insertEdge(const pair<uint32_t, uint32_t> & edge);  // insert an edge
    void deleteEdge(const pair<uint32_t, uint32_t> & edge);  // delete an edge
    vector<vector<pair<::uint32_t,::uint32_t>>> getDcoreDecomposition();  // get the dcore decomposition result



    // the kmax value and l_{max}(v, k) value of vertex
    vector<uint32_t> k_max;
    vector<vector<uint32_t>> l_max;

    //max k_max value of vertices in the graph
    uint32_t max_k_max;

    //the Dcore decomposition result
    vector<pair<uint32_t, uint32_t> > m_vResE;
};

#endif //DYNAMIC_DCORE_HINDEX_H
