
#ifndef DYNAMIC_DCORE_REPEEL_H
#define DYNAMIC_DCORE_REPEEL_H

#include "Common.h"

class Repeel final{
    private:
    typedef  struct final{
        uint32_t vid;
        uint32_t eid;
    } ArrayEntry;
    // data members
    uint32_t n_;  // the # of vertices
    uint32_t m_;  // the # of edges

    // the adjacency array representation
    vector<vector<ArrayEntry>> adj_in;
    vector<vector<ArrayEntry>> adj_out;

    // the set of edges
    vector<pair<uint32_t, uint32_t>> edges_;
    vector<uint32_t> nodes_;

    // the set of unique k-lists, or (k,0)-cores
    vector<vector<pair<uint32_t, uint32_t>>> unique_k_lists;


    void findNeib(vector<ArrayEntry> &vAdj1, vector<ArrayEntry> &vAdj2, vector<pair<uint32_t, uint32_t>> & tris);
    void completePartialDcore (vector<vector<pair<::uint32_t,uint32_t>>> & d_cores);

    bool optimize = true;  //whether optimize the baseline method

    public:
    Repeel(vector<pair<uint32_t, uint32_t> > & edges);
    Repeel(const Repeel&) = delete;
    Repeel& operator=(const Repeel&) = delete;

    vector<::uint32_t> in_coreness;


    void peelKlist(vector<vector<pair<uint32_t, uint32_t>>> & independent_k_lists);  // peel to get all unique k-lists
    void peelDcore(vector<vector<pair<uint32_t, uint32_t>>> & d_core_decomposition);  // peel unique k-lists to get dcore decomposition result

    // advanced peeling to get dcore decomposition result, TKDE19 supplementary material, b_insertion= true means insertion, otherwise deletion
    void optimizedPeelDcore(const vector<pair<::uint32_t,::uint32_t>>& modified_edge,
                            map<::uint32_t,uint32_t> &new_to_old_node_map,
                            bool b_insertion,
                            vector<vector<pair<uint32_t, uint32_t>>> & new_d_core_decomposition ,
                            vector<vector<pair<uint32_t, uint32_t>>> & old_d_core_decomposition);


    //the Dcore decomposition result
    vector<pair<uint32_t, uint32_t> > m_vResE;
};

#endif //DYNAMIC_DCORE_REPEEL_H
