

#ifndef DYNAMIC_DCORE_EDGEGENERATOR_H
#define DYNAMIC_DCORE_EDGEGENERATOR_H


#define abs_diff(x, y)  x > y ? x - y : y - x;
#include "Common.h"
#include <random>



class edgeGenerator {
private:
    typedef  struct final{
        uint32_t vid;
        uint32_t kmax;
    } ArrayEntry;
    struct GeneratedEdgeEntry{  //an edge (u,v) 
        uint32_t u;
        uint32_t v;
        int smaller_in_degree;
    } ;

    static bool in_degree_cmp(GeneratedEdgeEntry& x,  GeneratedEdgeEntry& y){
         return x.smaller_in_degree > y.smaller_in_degree;
    }

    std::vector<std::pair<uint32_t, uint32_t>> m_graph;

    std::set<uint32_t> node_recorder;

    ::uint32_t m_uiN;  // the # of vertices

    std::default_random_engine m_engine;

    bool isValidEdge(const std::pair<uint32_t, uint32_t>& edge) const;
    bool isValidVertex(uint32_t v) const;

public:
    std::vector<std::pair<uint32_t, uint32_t>> m_insert_edges, m_delete_edges;

    //map from new zero-based node id after deletion to old node id

    /**As long as we make sure that edge deletion/insertion do not cause vertex insertion/deletion,
     * we do not need to use the node map here.
    */
     std::map<::uint32_t,::uint32_t> new_to_old_node_map, old_to_new_node_map;

    edgeGenerator(const std::vector<std::pair<uint32_t, uint32_t>>& graph);
    ~edgeGenerator();

    void generatorInsertEdges(uint32_t numEdges, char* pcFile);
    void generatorDeleteEdges(uint32_t numEdges, char* pcFile);
    void generatorSubGraphs(vector<pair<uint32_t, uint32_t>> input_graph, string pcFile);

    static std::vector<vector<std::pair<uint32_t, uint32_t>>> getEdgeBatch(const std::vector<std::pair<uint32_t, uint32_t>> &edges_to_be_modified,
                   const  vector<vector<pair<::uint32_t,::uint32_t>>> & old_d_core_decomposition,
                   vector<pair<::uint32_t,::uint32_t>> & remaining_unbatched_edges,
                   const bool kmax_hierarchy, const bool kedge_set); //return the edge batch for parallel processing

    std::vector<std::pair<uint32_t, uint32_t>> getInsertedGraph(); //return the graph after edge insertion
    std::vector<std::pair<uint32_t, uint32_t>> getDeletedGraph();  //return the graph after edge deletion

    void getBiGraph(char* pcFile);
};

#endif //DYNAMIC_DCORE_EDGEGENERATOR_H
