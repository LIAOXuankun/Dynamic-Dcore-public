


#ifndef DYNAMIC_DCORE_IO_H
#define DYNAMIC_DCORE_IO_H

#include "Common.h"


// io class, for file read and write, the vertex of input file should be 0-based, and vertex id should be continuous

class Io final {
private:
    // edge array entry
    typedef struct final {
        uint32_t x;
        uint32_t y;
    } ArrayEntry;
    // order
    static bool cmp(const ArrayEntry &e1, const ArrayEntry &e2){
         if (e1.x != e2.x)
        {
            return e1.x < e2.x;
        }
        else
        {
            return e1.y < e2.y;
        }
    }
    // the set of edges <<edges> >
    //map<int, vector<pair<uint32_t, uint32_t> > > m_mpE;
    vector<ArrayEntry> m_vE;


public:
    uint32_t m_uiN;  // the # of vertices
    uint32_t m_uiM;  // the # of edges
    set<::uint32_t > node_counter;   //count number of vertices, record all the unique vertices having edges attached to them
    map<::uint32_t,::uint32_t> node_map;  //map the original vertex id to a continuous id， key: original id, value: continuous id

    void getEdges(vector<pair<uint32_t, uint32_t> > &vDesEdges);
    void readModifiedEdges(std::vector<std::pair<uint32_t, uint32_t>> &Edges, char* pcFile);
    void readFromFile(char* pcFile);
    void readFromFile(char* pcFile, vector<vector<pair<::uint32_t,::uint32_t>>> &d_core_decomposition);
    void writeToFile(char* pcFile, vector<vector<pair<::uint32_t,::uint32_t>>> &d_core_decomposition) const;
    void writeToFile(char* pcFile, vector<vector<uint32_t>> &l_max) const;
    void writeToFile(char* pcFile, vector<uint32_t> &k_max) const;
    void printDecomposition(vector<vector<pair<::uint32_t,::uint32_t>>> &d_core_decomposition) const;
    void printKlists(vector<vector<pair<::uint32_t,::uint32_t>>> &independent_k_lists) const;
};
#endif //DYNAMIC_DCORE_IO_H
